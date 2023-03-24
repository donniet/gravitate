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
    auto blocks = BlockStorage<int,int>::create_or_open(path, 10);
    *blocks.get(0) = -1;
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

    auto i = blocks.get(0);
    ASSERT_EQ(*i, -1);
}

TEST(BlockTest, WriteALot) {
    auto path = std::filesystem::temp_directory_path() / "block_test_int_int.blk";
    auto temp = temp_file(path); // RIAA to remove temp file

    std::cerr << "path: " << path << std::endl;

    auto blocks = BlockStorage<int,int>::create_or_open(path, 100);

    std::map<int,int> test_data;
    auto hasher = std::hash<int>();

    const int count = 1e4;

    for(int i = 0; i < count; i++) {
        // auto v = hasher(i);
        auto v = i;
        test_data[i] = v;
        *blocks.get(i) = v;
    }

    blocks.dump(std::cerr);

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

TEST(BlockTest, BigData) {
    auto path = std::filesystem::temp_directory_path() / "block_test_big_data.blk";
    auto temp = temp_file(path); // RIAA to remove temp file

    auto blocks = BlockStorage<BigData,int>::create_or_open(path, 1);

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
    ASSERT_TRUE(valid);
}