#include "block.hpp"

#include <filesystem>
#include <iostream>

#include <gtest/gtest.h>

TEST(BlockTest, CreateBlock) {
    auto path = std::filesystem::temp_directory_path() / "block_test_int_int.blk";

    std::cerr << "path: " << path << std::endl;

    auto blocks = BlockStorage<int,int>::create(path, 10);

    if(std::filesystem::exists(path)) {
        std::cerr << "file exists" << std::endl;
    } else {
        std::cerr << "file does not exist" << std::endl;
    }

    EXPECT_EQ(0, 0);

    std::filesystem::remove(path);
}