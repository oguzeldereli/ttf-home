// ttf::Tokenizer: HF tokenizer.json, the subset our models use (see ttf.h). Checked against HF `tokenizers` on a text
// set (tools/tokcheck.py): ids identical.
#include "ttf.h"

#include <cstdio>
#include <cstdlib>
#include <limits>
#include <stdexcept>

namespace ttf {

namespace {

// UTF-8 code point lengths of s (an invalid lead byte counts as one byte)
std::vector<std::string> utf8_chars(const std::string & s) {
    std::vector<std::string> out;
    for (size_t i = 0; i < s.size();) {
        const unsigned char c = (unsigned char) s[i];
        size_t              n = c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : (c >> 3) == 30 ? 4 : 1;
        n                     = std::min(n, s.size() - i);
        out.push_back(s.substr(i, n));
        i += n;
    }
    return out;
}

uint32_t utf8_cp(const std::string & ch) {
    const unsigned char * u = (const unsigned char *) ch.data();
    switch (ch.size()) {
        case 1: return u[0];
        case 2: return ((u[0] & 0x1F) << 6) | (u[1] & 0x3F);
        case 3: return ((u[0] & 0x0F) << 12) | ((u[1] & 0x3F) << 6) | (u[2] & 0x3F);
        case 4: return ((u[0] & 0x07) << 18) | ((u[1] & 0x3F) << 12) | ((u[2] & 0x3F) << 6) | (u[3] & 0x3F);
    }
    return 0;
}

void replace_all(std::string & s, const std::string & a, const std::string & b) {
    if (a.empty()) {
        return;
    }
    std::string out;
    size_t      i = 0;
    for (size_t j; (j = s.find(a, i)) != std::string::npos; i = j + a.size()) {
        out.append(s, i, j - i).append(b);
    }
    out.append(s, i, std::string::npos);
    s.swap(out);
}

uint64_t pair_key(int32_t a, int32_t b) {
    return ((uint64_t) (uint32_t) a << 32) | (uint32_t) b;
}

void add_norm(Tokenizer & t, const json & n) {
    if (n.is_null()) {
        return;
    }
    const std::string ty = n.at("type");
    if (ty == "Sequence") {
        for (const auto & e : n.at("normalizers")) {
            add_norm(t, e);
        }
    } else if (ty == "Prepend") {
        t.norm.push_back({ "Prepend", n.at("prepend"), "" });
    } else if (ty == "Replace" && n.at("pattern").contains("String")) {
        t.norm.push_back({ "Replace", n.at("pattern").at("String"), n.at("content") });
    } else {
        throw std::runtime_error("tokenizer: normalizer " + n.dump() + " not implemented");
    }
}

}  // namespace

std::unique_ptr<Tokenizer> Tokenizer::from_json(const std::string & tj, std::string * err) {
    try {
        auto         t = std::make_unique<Tokenizer>();
        const json   j = json::parse(tj);
        const json & m = j.at("model");
        if (m.at("type") != "BPE") {
            throw std::runtime_error("tokenizer: model " + m.at("type").get<std::string>() + " not implemented");
        }
        if (!m.value("dropout", json()).is_null() || !m.value("continuing_subword_prefix", json()).is_null() ||
            !m.value("end_of_word_suffix", json()).is_null()) {
            throw std::runtime_error("tokenizer: BPE dropout / subword prefix / suffix not implemented");
        }
        if (!j.value("pre_tokenizer", json()).is_null()) {
            throw std::runtime_error("tokenizer: pre_tokenizer " + j.at("pre_tokenizer").dump().substr(0, 200) +
                                     " not implemented");
        }
        add_norm(*t, j.value("normalizer", json()));
        int32_t mx = -1;
        for (auto it = m.at("vocab").begin(); it != m.at("vocab").end(); ++it) {
            t->vocab[it.key()] = it.value().get<int32_t>();
            mx                 = std::max(mx, it.value().get<int32_t>());
        }
        for (const auto & a : j.value("added_tokens", json::array())) {
            const int32_t id = a.at("id");
            t->vocab[a.at("content").get<std::string>()] = id;
            t->added.emplace_back(a.at("content").get<std::string>(), id);
            mx = std::max(mx, id);
        }
        std::sort(t->added.begin(), t->added.end(),
                  [](const auto & x, const auto & y) { return x.first.size() > y.first.size(); });
        t->id_to_tok.resize(mx + 1);
        for (const auto & [k, v] : t->vocab) {
            t->id_to_tok[v] = k;
        }
        int32_t rank = 0;
        for (const auto & e : m.at("merges")) {
            std::string a, b;
            if (e.is_string()) {
                const std::string s = e;
                const size_t      sp = s.find(' ', 1);  // the first token may itself be a space-free string; split at the first space after it
                a                    = s.substr(0, sp);
                b                    = s.substr(sp + 1);
            } else {
                a = e.at(0);
                b = e.at(1);
            }
            const int32_t ia = t->id_of(a), ib = t->id_of(b), im = t->id_of(a + b);
            if (ia < 0 || ib < 0 || im < 0) {
                throw std::runtime_error("tokenizer: merge '" + a + " " + b + "' outside the vocabulary");
            }
            t->merges.emplace(pair_key(ia, ib), std::make_pair(rank++, im));
        }
        t->byte_fallback = m.value("byte_fallback", false);
        t->fuse_unk      = m.value("fuse_unk", false);
        t->ignore_merges = m.value("ignore_merges", false);
        if (!m.value("unk_token", json()).is_null()) {
            t->unk = t->id_of(m.at("unk_token"));
        }
        for (int b = 0; b < 256; ++b) {
            char nm[8];
            std::snprintf(nm, sizeof(nm), "<0x%02X>", b);
            t->byte_id[b] = t->id_of(nm);
        }
        return t;
    } catch (const std::exception & e) {
        if (err) {
            *err = e.what();
        }
        return nullptr;
    }
}

int32_t Tokenizer::id_of(const std::string & tok) const {
    auto it = vocab.find(tok);
    return it == vocab.end() ? -1 : it->second;
}

// one pre-tokenized word: characters (byte fallback for unknown ones), then merges by rank, leftmost first
void Tokenizer::bpe_word(const std::string & w, std::vector<int32_t> & out) const {
    if (ignore_merges) {
        const int32_t id = id_of(w);
        if (id >= 0) {
            out.push_back(id);
            return;
        }
    }
    std::vector<int32_t> sym;
    for (const std::string & ch : utf8_chars(w)) {
        const int32_t id = id_of(ch);
        if (id >= 0) {
            sym.push_back(id);
            continue;
        }
        bool bytes_ok = byte_fallback;
        for (unsigned char c : ch) {
            bytes_ok = bytes_ok && byte_id[c] >= 0;
        }
        if (bytes_ok) {
            for (unsigned char c : ch) {
                sym.push_back(byte_id[c]);
            }
        } else if (unk >= 0) {
            if (!(fuse_unk && !sym.empty() && sym.back() == unk)) {
                sym.push_back(unk);
            }
        } else {
            throw std::runtime_error("tokenizer: character without a token and no unk");
        }
    }
    for (;;) {
        int32_t best = std::numeric_limits<int32_t>::max(), at = -1, nid = -1;
        for (size_t i = 0; i + 1 < sym.size(); ++i) {
            auto it = merges.find(pair_key(sym[i], sym[i + 1]));
            if (it != merges.end() && it->second.first < best) {
                best = it->second.first;
                at   = (int32_t) i;
                nid  = it->second.second;
            }
        }
        if (at < 0) {
            break;
        }
        sym[at] = nid;
        sym.erase(sym.begin() + at + 1);
    }
    out.insert(out.end(), sym.begin(), sym.end());
}

void Tokenizer::encode(const std::string & text, std::vector<std::string> & toks, std::vector<int32_t> & ids) const {
    ids.clear();
    // added tokens split the raw text first; each other piece is normalized and run through the model on its own
    size_t      i = 0;
    std::string piece;
    auto        flush = [&]() {
        if (piece.empty()) {
            return;
        }
        std::string s = piece;
        for (const NormStep & n : norm) {
            if (n.kind == "Prepend") {
                s = n.a + s;
            } else {
                replace_all(s, n.a, n.b);
            }
        }
        bpe_word(s, ids);
        piece.clear();
    };
    while (i < text.size()) {
        bool hit = false;
        for (const auto & [c, id] : added) {
            if (!c.empty() && text.compare(i, c.size(), c) == 0) {
                flush();
                ids.push_back(id);
                i += c.size();
                hit = true;
                break;
            }
        }
        if (!hit) {
            piece += text[i++];
        }
    }
    flush();
    toks.clear();
    for (int32_t id : ids) {
        toks.push_back(id_to_tok[id]);
    }
}

std::vector<int32_t> Tokenizer::encode(const std::string & text) const {
    std::vector<std::string> t;
    std::vector<int32_t>     ids;
    encode(text, t, ids);
    return ids;
}

std::string Tokenizer::decode(const std::vector<int32_t> & ids, bool strip_prepend) const {
    std::string out;
    for (int32_t id : ids) {
        if (id < 0 || id >= (int32_t) id_to_tok.size()) {
            continue;
        }
        const std::string & t = id_to_tok[id];
        if (byte_fallback && t.size() == 6 && t.compare(0, 3, "<0x") == 0 && t[5] == '>') {
            out += (char) std::strtol(t.substr(3, 2).c_str(), nullptr, 16);
            continue;
        }
        bool special = false;
        for (const auto & [c, aid] : added) {
            special = special || aid == id;
        }
        if (!special) {
            out += t;
        }
    }
    for (auto it = norm.rbegin(); it != norm.rend(); ++it) {
        if (it->kind == "Replace") {
            replace_all(out, it->b, it->a);
        } else if (it->kind == "Prepend" && strip_prepend) {
            std::string pre = it->a;  // as it reads after the later steps were undone
            for (auto j = norm.rbegin(); j != it; ++j) {
                if (j->kind == "Replace") {
                    replace_all(pre, j->b, j->a);
                }
            }
            if (out.compare(0, pre.size(), pre) == 0) {
                out.erase(0, pre.size());
            }
        }
    }
    return out;
}

std::vector<int32_t> Tokenizer::encode_multichar_cjk(const std::string & text) const {
    std::vector<std::string> toks;
    std::vector<int32_t>     ids, out;
    encode(text, toks, ids);
    for (size_t k = 0; k < toks.size(); ++k) {
        std::string clean = toks[k];
        replace_all(clean, "\xE2\x96\x81", "");  // U+2581
        const auto chars = utf8_chars(clean);
        bool       cjk   = chars.size() >= 2 && vocab.count(clean);
        for (const std::string & ch : chars) {
            const uint32_t cp = utf8_cp(ch);
            cjk               = cjk && cp >= 0x4E00 && cp <= 0x9FFF;
        }
        if (!cjk) {
            out.push_back(ids[k]);
            continue;
        }
        for (const std::string & ch : chars) {
            const int32_t id = id_of(ch);
            out.push_back(id >= 0 ? id : unk);
        }
    }
    return out;
}

}  // namespace ttf
