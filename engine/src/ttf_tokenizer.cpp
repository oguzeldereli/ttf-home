// ttf::Tokenizer: HF tokenizer.json, the subset our models use (see ttf.h). Checked against HF `tokenizers` on a text
// set (tools/tokcheck.py): ids identical.
#include "ttf.h"

#include <algorithm>
#include <cctype>
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

#include "ttf_unicode.inc"

template <size_t W> const uint32_t * range_find(const uint32_t (*tab)[W], size_t n, uint32_t cp) {
    size_t lo = 0, hi = n;
    while (lo < hi) {
        const size_t mid = (lo + hi) / 2;
        if (tab[mid][1] < cp) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return lo < n && tab[lo][0] <= cp && cp <= tab[lo][1] ? tab[lo] : nullptr;
}

uint32_t uflags(uint32_t cp) {
    const uint32_t * e = range_find(k_uclass, sizeof(k_uclass) / sizeof(k_uclass[0]), cp);
    return e ? e[2] : 0;
}

bool uspace(uint32_t cp) {
    return std::binary_search(std::begin(k_uspace), std::end(k_uspace), cp);
}

int uccc(uint32_t cp) {
    if (cp < 0x300) {
        return 0;
    }
    const uint32_t * e = range_find(k_uccc, sizeof(k_uccc) / sizeof(k_uccc[0]), cp);
    return e ? (int) e[2] : 0;
}

std::vector<uint32_t> to_cps(const std::string & s) {
    std::vector<uint32_t> out;
    for (const std::string & ch : utf8_chars(s)) {
        out.push_back(utf8_cp(ch));
    }
    return out;
}

void put_utf8(std::string & o, uint32_t cp) {
    if (cp < 0x80) {
        o += (char) cp;
    } else if (cp < 0x800) {
        o += (char) (0xC0 | (cp >> 6));
        o += (char) (0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
        o += (char) (0xE0 | (cp >> 12));
        o += (char) (0x80 | ((cp >> 6) & 0x3F));
        o += (char) (0x80 | (cp & 0x3F));
    } else {
        o += (char) (0xF0 | (cp >> 18));
        o += (char) (0x80 | ((cp >> 12) & 0x3F));
        o += (char) (0x80 | ((cp >> 6) & 0x3F));
        o += (char) (0x80 | (cp & 0x3F));
    }
}

constexpr uint32_t SB = 0xAC00, LB = 0x1100, VB = 0x1161, TB = 0x11A7, LC = 19, VC = 21, TC = 28, NC = VC * TC,
                   SC = LC * NC;

// NFC (UAX #15): full canonical decomposition, canonical ordering, canonical composition
std::string nfc(const std::string & s) {
    bool low = true;
    for (unsigned char c : s) {
        low = low && c < 0xCC;  // every code point below U+0300 (lead bytes < 0xCC)
    }
    if (low) {
        return s;
    }
    std::vector<uint32_t> d;
    for (uint32_t cp : to_cps(s)) {
        if (cp >= SB && cp < SB + SC) {
            const uint32_t i = cp - SB;
            d.push_back(LB + i / NC);
            d.push_back(VB + (i % NC) / TC);
            if (i % TC) {
                d.push_back(TB + i % TC);
            }
            continue;
        }
        const size_t nd = sizeof(k_udecomp) / sizeof(k_udecomp[0]);
        const auto * e  = std::lower_bound(k_udecomp, k_udecomp + nd, cp,
                                           [](const uint32_t (&r)[5], uint32_t v) { return r[0] < v; });
        if (e != k_udecomp + nd && (*e)[0] == cp) {
            for (int k = 1; k < 5 && (*e)[k]; ++k) {
                d.push_back((*e)[k]);
            }
        } else {
            d.push_back(cp);
        }
    }
    for (size_t i = 1; i < d.size(); ++i) {  // stable insertion sort of each run of non-starters by class
        const int c = uccc(d[i]);
        if (c == 0) {
            continue;
        }
        for (size_t j = i; j > 0 && uccc(d[j - 1]) > c; --j) {
            std::swap(d[j - 1], d[j]);
        }
    }
    std::vector<uint32_t> r;
    long                  si      = -1;
    int                   last_cc = 0;
    for (uint32_t c : d) {
        const int cc = uccc(c);
        if (si >= 0 && ((long) r.size() - 1 == si || (last_cc != 0 && last_cc < cc))) {
            const uint32_t st   = r[si];
            uint32_t       comp = 0;
            if (st >= LB && st < LB + LC && c >= VB && c < VB + VC) {
                comp = SB + ((st - LB) * VC + (c - VB)) * TC;
            } else if (st >= SB && st < SB + SC && (st - SB) % TC == 0 && c > TB && c < TB + TC) {
                comp = st + (c - TB);
            } else {
                const size_t nc = sizeof(k_ucomp) / sizeof(k_ucomp[0]);
                const auto * e  = std::lower_bound(k_ucomp, k_ucomp + nc, std::make_pair(st, c),
                                                   [](const uint32_t (&x)[3], const std::pair<uint32_t, uint32_t> & v) {
                                                       return x[0] < v.first || (x[0] == v.first && x[1] < v.second);
                                                   });
                if (e != k_ucomp + nc && (*e)[0] == st && (*e)[1] == c) {
                    comp = (*e)[2];
                }
            }
            if (comp) {
                r[si] = comp;
                continue;
            }
        }
        if (cc == 0) {
            si = (long) r.size();
        }
        r.push_back(c);
        last_cc = cc;
    }
    std::string o;
    for (uint32_t cp : r) {
        put_utf8(o, cp);
    }
    return o;
}

const char * k_re_llama3 =
    "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|"
    "\\s+(?!\\S)|\\s+";
const char * k_re_qwen35 =
    "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|"
    "\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";

// GPT-2's regex (the ByteLevel pre-tokenizer's, use_regex): the end of its match at i, or i when nothing matches
//   's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
size_t gpt2_match(const std::vector<uint32_t> & c, size_t i) {
    const size_t n = c.size();
    auto         L = [](uint32_t x) { return (uflags(x) & 1u) != 0; };
    auto         N = [](uint32_t x) { return (uflags(x) & 4u) != 0; };
    if (c[i] == '\'' && i + 1 < n) {
        const uint32_t a = c[i + 1];
        if (a == 's' || a == 't' || a == 'm' || a == 'd') {
            return i + 2;
        }
        if (i + 2 < n && ((a == 'r' && c[i + 2] == 'e') || (a == 'v' && c[i + 2] == 'e') || (a == 'l' && c[i + 2] == 'l'))) {
            return i + 3;
        }
    }
    auto run = [&](auto cls) -> size_t {  // ' '?cls+ at i, or i
        size_t j = i + (c[i] == ' ' && i + 1 < n && cls(c[i + 1]) ? 1 : 0);
        if (!cls(c[j])) {
            return i;
        }
        while (j < n && cls(c[j])) {
            ++j;
        }
        return j;
    };
    for (size_t e : { run(L), run(N), run([&](uint32_t x) { return !uspace(x) && !L(x) && !N(x); }) }) {
        if (e > i) {
            return e;
        }
    }
    if (!uspace(c[i])) {
        return i;
    }
    size_t re = i;
    while (re < n && uspace(c[re])) {
        ++re;
    }
    return re == n || re - i < 2 ? re : re - 1;  // \s+(?!\S), else \s+
}

bool hf_punct(uint32_t cp) {  // tokenizers' is_punc: ASCII punctuation or a Unicode P* character
    return (cp < 0x80 && std::ispunct((int) cp)) || (uflags(cp) & 8u) != 0;
}

// the end of the regex's match at i (leftmost alternative first, as the regex engine), or i when nothing matches.
// split 1: Llama-3 (letters \p{L}, numbers 1-3); 2: Qwen3.5 (letters [\p{L}\p{M}], numbers 1, marks out of alt 4)
size_t split_match(const std::vector<uint32_t> & c, size_t i, int split) {
    const size_t n   = c.size();
    auto         L   = [&](uint32_t x) { return (uflags(x) & (split == 2 ? 3u : 1u)) != 0; };
    auto         Lp  = [&](uint32_t x) { return (uflags(x) & 1u) != 0; };
    auto         N   = [&](uint32_t x) { return (uflags(x) & 4u) != 0; };
    auto         nl  = [](uint32_t x) { return x == '\r' || x == '\n'; };
    auto         low = [](uint32_t x) { return x == 0x17F ? (uint32_t) 's' : x < 0x80 ? (uint32_t) std::tolower((int) x) : x; };
    // 1: (?i:'s|'t|'re|'ve|'m|'ll|'d)
    if (c[i] == '\'' && i + 1 < n) {
        const uint32_t a = low(c[i + 1]);
        if (a == 's' || a == 't' || a == 'm' || a == 'd') {
            return i + 2;
        }
        if (i + 2 < n) {
            const uint32_t b = low(c[i + 2]);
            if ((a == 'r' && b == 'e') || (a == 'v' && b == 'e') || (a == 'l' && b == 'l')) {
                return i + 3;
            }
        }
    }
    // 2: [^\r\n\p{L}\p{N}]?LETTERS+
    if (!nl(c[i]) && !Lp(c[i]) && !N(c[i]) && i + 1 < n && L(c[i + 1])) {
        size_t j = i + 1;
        while (j < n && L(c[j])) {
            ++j;
        }
        return j;
    }
    if (L(c[i])) {
        size_t j = i;
        while (j < n && L(c[j])) {
            ++j;
        }
        return j;
    }
    // 3: \p{N}{1,3} / \p{N}
    if (N(c[i])) {
        size_t j = i;
        while (j < n && j - i < (split == 2 ? 1u : 3u) && N(c[j])) {
            ++j;
        }
        return j;
    }
    // 4: ' '?[^\s LETTERS-or-L \p{N}]+[\r\n]*
    auto P = [&](uint32_t x) { return !uspace(x) && !L(x) && !Lp(x) && !N(x); };
    {
        size_t j = i + (c[i] == ' ' && i + 1 < n && P(c[i + 1]) ? 1 : 0);
        if (j < n && P(c[j])) {
            while (j < n && P(c[j])) {
                ++j;
            }
            while (j < n && nl(c[j])) {
                ++j;
            }
            return j;
        }
    }
    if (!uspace(c[i])) {
        return i;
    }
    size_t re = i;
    while (re < n && uspace(c[re])) {
        ++re;
    }
    // 5: \s*[\r\n]+ : up to the last line break of the whitespace run
    for (size_t k = re; k > i; --k) {
        if (nl(c[k - 1])) {
            return k;
        }
    }
    // 6: \s+(?!\S) ; 7: \s+
    if (re == n || re - i < 2) {
        return re;
    }
    return re - 1;
}

// GPT-2's byte -> char map
const std::vector<uint32_t> & byte_chars() {
    static const std::vector<uint32_t> m = [] {
        std::vector<uint32_t> t(256);
        uint32_t              extra = 256;
        for (uint32_t b = 0; b < 256; ++b) {
            const bool keep = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255);
            t[b]            = keep ? b : extra++;
        }
        return t;
    }();
    return m;
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
    } else if (ty == "NFC") {
        t.norm.push_back({ "NFC", "", "" });
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
        auto empty = [&](const char * k) {
            const json v = m.value(k, json());
            return v.is_null() || (v.is_string() && v.get<std::string>().empty());
        };
        if (!m.value("dropout", json()).is_null() || !empty("continuing_subword_prefix") || !empty("end_of_word_suffix")) {
            throw std::runtime_error("tokenizer: BPE dropout / subword prefix / suffix not implemented");
        }
        const json pre = j.value("pre_tokenizer", json());
        if (!pre.is_null()) {
            const json seq = pre.at("type") == "Sequence" ? pre.at("pretokenizers") : json::array({ pre });
            for (const json & p : seq) {
                const std::string ty = p.at("type");
                if (ty == "Punctuation" && p.value("behavior", std::string()) == "Contiguous" && !t->punct &&
                    t->split == 0 && !t->byte_level && !t->digits) {
                    t->punct = true;  // only first in the sequence: applied before the split
                } else if (ty == "Digits" && p.value("individual_digits", false) && !t->digits) {
                    t->digits = true;  // applied last (after the byte-level map): nothing may follow it
                } else if (t->digits) {
                    throw std::runtime_error("tokenizer: pre_tokenizer after Digits not implemented");
                } else if (ty == "ByteLevel" && !t->byte_level && t->split == 0 && !p.value("add_prefix_space", false) &&
                           p.value("use_regex", true)) {
                    t->byte_level = true;
                    t->split      = 3;
                } else if (ty == "Split" && t->split == 0 && !t->byte_level && p.at("pattern").contains("Regex") &&
                    p.value("behavior", std::string()) == "Isolated" && !p.value("invert", false)) {
                    const std::string re = p.at("pattern").at("Regex");
                    t->split             = re == k_re_llama3 ? 1 : re == k_re_qwen35 ? 2 : 0;
                    if (!t->split) {
                        throw std::runtime_error("tokenizer: Split regex " + re + " not implemented");
                    }
                } else if (ty == "ByteLevel" && !t->byte_level && !p.value("add_prefix_space", false) &&
                           !p.value("use_regex", true)) {
                    t->byte_level = true;
                } else {
                    throw std::runtime_error("tokenizer: pre_tokenizer " + p.dump().substr(0, 300) + " not implemented");
                }
            }
        }
        const json dec = j.value("decoder", json());
        if (t->byte_level != (!dec.is_null() && dec.value("type", std::string()) == "ByteLevel")) {
            throw std::runtime_error("tokenizer: a ByteLevel pre-tokenizer without its decoder (or the reverse)");
        }
        // the post-processor's special tokens around a single sequence (TemplateProcessing, possibly in a Sequence)
        const json pp = j.value("post_processor", json());
        if (!pp.is_null()) {
            const json seq = pp.at("type") == "Sequence" ? pp.at("processors") : json::array({ pp });
            for (const json & p : seq) {
                if (p.at("type") != "TemplateProcessing") {
                    continue;  // ByteLevel / offsets-only processors change no ids
                }
                bool seen_a = false;
                for (const json & e : p.at("single")) {
                    if (e.contains("Sequence")) {
                        seen_a = true;
                    } else {
                        const std::string sp = e.at("SpecialToken").at("id");
                        const json &      st = p.at("special_tokens").at(sp).at("ids");
                        for (const auto & id : st) {
                            (seen_a ? t->post_post : t->post_pre).push_back(id.get<int32_t>());
                        }
                    }
                }
            }
        }
        add_norm(*t, j.value("normalizer", json()));
        int32_t mx = -1;
        for (auto it = m.at("vocab").begin(); it != m.at("vocab").end(); ++it) {
            t->vocab[it.key()] = it.value().get<int32_t>();
            mx                 = std::max(mx, it.value().get<int32_t>());
        }
        for (const auto & a : j.value("added_tokens", json::array())) {
            const int32_t id = a.at("id");
            if (a.value("lstrip", false) || a.value("rstrip", false) || a.value("single_word", false)) {
                throw std::runtime_error("tokenizer: added token " + a.dump() + ": lstrip / rstrip / single_word not implemented");
            }
            t->vocab[a.at("content").get<std::string>()] = id;
            (a.value("normalized", false) ? t->added_norm : t->added).emplace_back(a.at("content").get<std::string>(), id);
            if (a.value("special", false)) {
                t->special_ids.push_back(id);
            }
            mx = std::max(mx, id);
        }
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
        for (auto & [c, id] : t->added_norm) {  // their content as the normalizer leaves it
            c = t->normalize(c);
        }
        for (auto * v : { &t->added, &t->added_norm }) {
            std::stable_sort(v->begin(), v->end(), [](const auto & x, const auto & y) { return x.first.size() > y.first.size(); });
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

std::string Tokenizer::normalize(const std::string & text) const {
    std::string s = text;
    for (const NormStep & n : norm) {
        if (n.kind == "NFC") {
            s = nfc(s);
        } else if (n.kind == "Prepend") {
            s = n.a + s;
        } else {
            replace_all(s, n.a, n.b);
        }
    }
    return s;
}

// an already normalized piece: pre-tokenizer, byte-level map, BPE
void Tokenizer::model_piece(const std::string & s, std::vector<int32_t> & ids) const {
    // Punctuation (Contiguous): runs of punctuation and the runs between them, in order
    std::vector<std::string> pieces;
    if (punct) {
        std::string cur;
        int         kind = -1;
        for (uint32_t cp : to_cps(s)) {
            const int k = hf_punct(cp) ? 1 : 0;
            if (k != kind && !cur.empty()) {
                pieces.push_back(cur);
                cur.clear();
            }
            kind = k;
            put_utf8(cur, cp);
        }
        if (!cur.empty()) {
            pieces.push_back(cur);
        }
    } else {
        pieces.push_back(s);
    }
    std::vector<std::string> words;
    for (const std::string & ps : pieces) {
        if (split) {
            const std::vector<uint32_t> c = to_cps(ps);
            std::string                 gap;
            for (size_t i = 0; i < c.size();) {
                const size_t e = split == 3 ? gpt2_match(c, i) : split_match(c, i, split);
                if (e == i) {  // no alternative matches here: the character joins the gap piece (Isolated keeps gaps)
                    put_utf8(gap, c[i++]);
                    continue;
                }
                if (!gap.empty()) {
                    words.push_back(gap);
                    gap.clear();
                }
                std::string w;
                for (size_t k = i; k < e; ++k) {
                    put_utf8(w, c[k]);
                }
                words.push_back(w);
                i = e;
            }
            if (!gap.empty()) {
                words.push_back(gap);
            }
        } else {
            words.push_back(ps);
        }
    }
    for (const std::string & w : words) {
        std::string m;
        if (byte_level) {
            for (unsigned char b : w) {
                put_utf8(m, byte_chars()[b]);
            }
        } else {
            m = w;
        }
        if (!digits) {
            bpe_word(m, ids);
            continue;
        }
        // Digits (individual): each numeric character of the mapped word its own piece
        std::string cur;
        for (uint32_t cp : to_cps(m)) {
            if ((uflags(cp) & 4u) != 0) {
                if (!cur.empty()) {
                    bpe_word(cur, ids);
                    cur.clear();
                }
                std::string d;
                put_utf8(d, cp);
                bpe_word(d, ids);
            } else {
                put_utf8(cur, cp);
            }
        }
        if (!cur.empty()) {
            bpe_word(cur, ids);
        }
    }
}

namespace {
// text split at the listed tokens (longest first at each position): matches -> their id, the gaps -> gap()
template <typename F>
void split_added(const std::string & text, const std::vector<std::pair<std::string, int32_t>> & toks,
                 std::vector<int32_t> & ids, F && gap) {
    size_t i = 0, g = 0;
    while (i < text.size()) {
        bool hit = false;
        for (const auto & [c, id] : toks) {
            if (!c.empty() && text.compare(i, c.size(), c) == 0) {
                if (i > g) {
                    gap(text.substr(g, i - g));
                }
                ids.push_back(id);
                i += c.size();
                g   = i;
                hit = true;
                break;
            }
        }
        if (!hit) {
            ++i;
        }
    }
    if (text.size() > g) {
        gap(text.substr(g));
    }
}
}  // namespace

void Tokenizer::encode(const std::string & text, std::vector<std::string> & toks, std::vector<int32_t> & ids) const {
    ids.clear();
    // HF order: the non-normalized added tokens split the raw text; each gap is normalized, split at the normalized
    // added tokens, and what is left goes through the pre-tokenizer and the model
    split_added(text, added, ids, [&](const std::string & piece) {
        split_added(normalize(piece), added_norm, ids, [&](const std::string & s) { model_piece(s, ids); });
    });
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
        if (std::find(special_ids.begin(), special_ids.end(), id) == special_ids.end()) {
            out += t;
        }
    }
    if (byte_level) {  // chars back to bytes (the specials were dropped above)
        std::vector<int> inv(512, -1);
        for (int b = 0; b < 256; ++b) {
            inv[byte_chars()[b]] = b;
        }
        std::string bytes;
        for (uint32_t cp : to_cps(out)) {
            if (cp < inv.size() && inv[cp] >= 0) {
                bytes += (char) inv[cp];
            } else {
                put_utf8(bytes, cp);
            }
        }
        return bytes;
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
