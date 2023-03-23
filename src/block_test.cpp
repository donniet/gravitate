#include "block.hpp"

#include <filesystem>
#include <iostream>

#include <gtest/gtest.h>

void create_block_file(string const & path) {
    auto blocks = BlockStorage<int,int>::create_or_open(path, 10);
    blocks.get(0) = -1;
    blocks.save_one(0);

    blocks.dump(std::cerr);
}

TEST(BlockTest, CreateBlock) {
    auto path = std::filesystem::temp_directory_path() / "block_test_int_int.blk";

    std::cerr << "path: " << path << std::endl;

    create_block_file(path);

    ASSERT_TRUE(std::filesystem::exists(path));

    auto blocks = BlockStorage<int,int>::create_or_open(path, 10);
    blocks.dump(std::cerr);

    int & i = blocks.get(0);
    ASSERT_EQ(i, -1);



    EXPECT_EQ(0, 0);

    std::filesystem::remove(path);
}