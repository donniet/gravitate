#include "block.hpp"

#include <filesystem>
#include <iostream>

#include <gtest/gtest.h>

struct temp_file {
    std::string file_name;
    temp_file(string const & file_name) : file_name(file_name) {
        std::filesystem::remove(file_name);
    }
    ~temp_file() {
        std::filesystem::remove(file_name);
    }
};

void create_block_file(string const & path) {
    auto blocks = BlockStorage<int,int>::create_or_open(path, 10);
    blocks.get(0) = -1;
    blocks.save_one(0);

    blocks.dump(std::cerr);
}

TEST(BlockTest, CreateBlock) {
    auto path = std::filesystem::temp_directory_path() / "block_test_int_int.blk";
    auto temp = temp_file(path); // RIAA to remove temp file

    std::cerr << "path: " << path << std::endl;

    ASSERT_FALSE(std::filesystem::exists(path));

    create_block_file(path);

    ASSERT_TRUE(std::filesystem::exists(path));

    auto blocks = BlockStorage<int,int>::create_or_open(path, 10);
    blocks.dump(std::cerr);

    int & i = blocks.get(0);
    ASSERT_EQ(i, -1);
}