// ttf-tok: the .ttf file's tokenizer on stdin lines -> one line of comma-separated ids per input line (no weights loaded).
//   ttf-tok MODEL [--post mask_multichar_cjk]   (default post: the IR driver's tokenizer_post)
#include "ttf.h"

#include <cstdio>
#include <iostream>
#include <string>

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s MODEL [--post NAME] < lines\n", argv[0]);
        return 1;
    }
    gguf_init_params gp = { true, nullptr };
    gguf_context *   g  = gguf_init_from_file(argv[1], gp);
    if (!g) {
        std::fprintf(stderr, "cannot read %s\n", argv[1]);
        return 1;
    }
    auto kv = [&](const char * k) -> std::string {
        const int64_t i = gguf_find_key(g, k);
        return i < 0 ? std::string() : std::string(gguf_get_val_str(g, i));
    };
    std::string post;
    const std::string irs = kv("ttf.ir");
    if (!irs.empty()) {
        const auto ir = nlohmann::json::parse(irs);
        if (ir.contains("driver")) {
            post = ir.at("driver").value("attr", nlohmann::json::object()).value("tokenizer_post", std::string());
        }
    }
    for (int i = 2; i + 1 < argc; ++i) {
        if (std::string(argv[i]) == "--post") {
            post = argv[++i];
        }
    }
    std::string err;
    auto        tok = ttf::Tokenizer::from_json(kv("ttf.tokenizer.json"), &err);
    gguf_free(g);
    if (!tok) {
        std::fprintf(stderr, "tokenizer: %s\n", err.c_str());
        return 1;
    }
    for (std::string line; std::getline(std::cin, line);) {
        const auto ids = post == "mask_multichar_cjk" ? tok->encode_multichar_cjk(line) : tok->encode(line);
        for (size_t k = 0; k < ids.size(); ++k) {
            std::printf(k ? ",%d" : "%d", ids[k]);
        }
        std::printf("\n");
    }
    return 0;
}
