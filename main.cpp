#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "csrc/modeling.hpp"
#include "csrc/tokenizer.hpp"
#include "csrc/weights.hpp"

namespace {

struct Args {
    std::string command;
    std::string prompt;
    std::string weights_dir = "granite-3.0-3b-a800m-instruct";
    int max_tokens = 32;
    float temperature = 0.8F;
    int top_k = 40;
    uint64_t seed = 42;
};

void usage() {
    std::printf(
        "usage:\n"
        "  granite generate \"<prompt>\" \"<weights_dir>\"\n"
        "  granite generate \"<prompt>\" \"<weights_dir>\" Max-Tokens Temperature Top-K Seed\n");
}

void init_args(int argc, char** argv, Args& args) {
    if (argc < 3) return;
    args.command = argv[1];
    args.prompt = argv[2];
    if (argc >= 4) args.weights_dir = argv[3];
    if (argc >= 5) args.max_tokens = std::atoi(argv[4]);
    if (argc >= 6) args.temperature = std::atof(argv[5]);
    if (argc >= 7) args.top_k = std::atoi(argv[6]);
    if (argc >= 8) args.seed = static_cast<uint64_t>(std::atoi(argv[7]));
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        usage();
        return EXIT_FAILURE;
    }

    Args args;
    init_args(argc, argv, args);

    try {
        const std::string config_path = args.weights_dir + "/config.json";
        const std::string tokenizer_path = args.weights_dir + "/vocab.json";
        const std::string merges_path = args.weights_dir + "/merges.txt";

        std::fprintf(stderr, "loading tokenizer...\n");
        tk::Tokenizer tokenizer(tokenizer_path, merges_path);

        std::fprintf(stderr, "loading Granite config...\n");
        const granite::GraniteConfig config = granite::read_config(config_path);

        std::fprintf(stderr, "loading weights from %s...\n", args.weights_dir.c_str());
        granite::GraniteWeights weights = granite::load_granite_weights(args.weights_dir, config);
        granite::GraniteModel model(std::move(weights));

        if (args.command == "generate") {
            const std::string output = model.generate(tokenizer, args.prompt,
                                                     args.max_tokens,
                                                     args.temperature,
                                                     args.top_k,
                                                     args.seed);
            std::printf("%s\n", output.c_str());
            return 0;
        }

        std::fprintf(stderr, "unsupported command: %s\n", args.command.c_str());
        return EXIT_FAILURE;
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "error: %s\n", ex.what());
        return EXIT_FAILURE;
    }
}
