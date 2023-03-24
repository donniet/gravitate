#include "block.hpp"

#include <filesystem>
#include <iostream>
#include <functional>
#include <array>
#include <random>

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
    BlockStorage<int,int> blocks(path, 10);
    *blocks.get(0) = -1;
    blocks.save_one(0);
}

TEST(BlockTest, CreateAndLoadBlock) {
    auto path = std::filesystem::temp_directory_path() / "block_test_int_int.blk";
    auto temp = temp_file(path); // RIAA to remove temp file

    std::cerr << "path: " << path << std::endl;

    auto pblocks = std::make_shared<BlockStorage<int,int>>(path, 10);
    { *pblocks->get(0) = -1; }
    pblocks = nullptr;

    pblocks = std::make_shared<BlockStorage<int,int>>(path, 10);
    int i = *pblocks->get(0);

    ASSERT_EQ(i, -1);
}

TEST(BlockTest, CreateBlock) {
    auto path = std::filesystem::temp_directory_path() / "block_test_int_int.blk";
    auto temp = temp_file(path); // RIAA to remove temp file

    std::cerr << "path: " << path << std::endl;

    ASSERT_FALSE(std::filesystem::exists(path));

    create_block_file(path);

    ASSERT_TRUE(std::filesystem::exists(path));

    BlockStorage<int,int> blocks(path, 10);
    blocks.dump(std::cerr);

    auto i = blocks.get(0);

    std::cerr << "file size: " << std::filesystem::file_size(path) << std::endl;
    ASSERT_EQ(*i, -1);
}

TEST(BlockTest, WriteALot) {
    auto path = std::filesystem::temp_directory_path() / "block_test_int_int.blk";
    auto temp = temp_file(path); // RIAA to remove temp file

    std::cerr << "path: " << path << std::endl;

    BlockStorage<int,int> blocks(path, 100);

    std::map<int,int> test_data;
    auto hasher = std::hash<int>();

    const int count = 1e4;

    for(int i = 0; i < count; i++) {
        // auto v = hasher(i);
        auto v = i;
        test_data[i] = v;
        *blocks.get(i) = v;
    }

    // verify
    bool valid = true;
    int i = 0;
    for(; i < count; i++) {
        auto v = blocks.get(i);
        if(test_data[i] != *v) {
            valid = false;
            break;
        }
    }
    std::cerr << "count correct: " << i << std::endl;
    std::cerr << "file size: " << std::filesystem::file_size(path) << std::endl;

    ASSERT_TRUE(valid);
}

struct BigData {
    std::array<int,1024> data;
    BigData(int i) {
        data[0] = i;
    }
    BigData() {}
};

std::ostream & operator<<(std::ostream & os, const BigData & v) {
    return os << v.data[0];
}

template<> struct FixedReadWriter<BigData> {
    void read(std::istream & is, BigData & v) {
        is.read(reinterpret_cast<char *>(&v.data[0]), sizeof(int) * 1024);
    }
    void write(std::ostream & os, const BigData & v) {
        os.write(reinterpret_cast<const char *>(&v.data[0]), sizeof(int) * 1024);
    }
};

TEST(BlockTest, BigDataRandomKeys) {
    auto path = std::filesystem::temp_directory_path() / "block_test_big_data.blk";
    auto temp = temp_file(path); // RIAA to remove temp file

    BlockStorage<BigData,int> blocks(path, 100);

    size_t count = 1e4;
    std::map<int,int> test_data;
    auto hasher = std::hash<int>();
    for(int i = 0; i < count; i++) {
        auto h = hasher(i);
        auto v = BigData(i);
        test_data[h] = i;
        *blocks.get(h) = v;
    }

    // verify
    bool valid = true;
    int i = 0;
    for(; i < count; i++) {
        auto v = blocks.get(i);
        if(test_data[i] != v->data[0]) {
            valid = false;
            break;
        }
    }

    ASSERT_TRUE(valid);
}

TEST(BlockTest, BigData) {
    auto path = std::filesystem::temp_directory_path() / "block_test_big_data.blk";
    auto temp = temp_file(path); // RIAA to remove temp file

    BlockStorage<BigData,int> blocks(path, 1);

    size_t count = 1e4;
    for(int i = 0; i < count; i++) {
        *blocks.get(i) = BigData(i);
    }

    // verify
    bool valid = true;
    int i = 0;
    for(; i < count; i++) {
        auto v = blocks.get(i);
        if(v->data[0] != i) {
            valid = false;
            break;
        }
    }
    std::cerr << "count correct: " << i << std::endl;
    std::cerr << "file size: " << std::filesystem::file_size(path) << std::endl;

    ASSERT_TRUE(valid);
}