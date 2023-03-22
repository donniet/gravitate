#pragma once

#include <iostream>
#include <fstream>
#include <filesystem>
#include <memory>
#include <string>
#include <map>
#include <cstdio>
#include <exception>
#include <sstream>


using std::istream;
using std::ostream;
using std::fstream;
using std::string;

template<typename T>
struct FixedReadWriter {
    void read(istream & is, T & data);
    void write(ostream & os, T const & data);
};

template<> struct FixedReadWriter<int> {
    void read(istream & is, int & data) {
        is.read(reinterpret_cast<char *>(&data), sizeof(int));
    }
    void write(ostream & os, int const & data) {
        os.write(reinterpret_cast<const char *>(&data), sizeof(int));
    }
};

// Block and Key should be default constructable
// Blocks and Keys must all serialize to the same bytesize
// Keys can never be destroyed
template<typename Block, typename Key>
class BlockStorage {
public:
    Block & get(Key const &key);
    void save_one(Key const &key, Block const &block);
    // saves everything to disk
    void save_all();

    static BlockStorage<Block, Key> load_from_file(string const & file_name, size_t maximum_loaded_blocks);
    static BlockStorage<Block, Key> create(string const & file_name, size_t maximum_loaded_blocks);

    BlockStorage(BlockStorage<Block, Key> const &);
    BlockStorage(BlockStorage<Block, Key> &&) noexcept;
private:
    BlockStorage();

    void remove_one();
    void double_storage();
    size_t grow_index(Key const & key);
    void open_file();
    void close_file();

    std::map<Key,size_t> index_;
    size_t next_index_;

    std::map<Key,std::shared_ptr<Block>> loaded_;
    
    std::deque<Key> keys_;
    size_t maximum_loaded_blocks_; // how many blocks to have loaded in memory at a time
    std::fstream block_file_;
    std::string path_;

    size_t index_size_; // block_file_[N - index_size_, N) will contain the index
    size_t data_size_; // block_file_[0, data_size_) will containe the blocks
    size_t file_size_;

    size_t cache_hit_;
    size_t cache_miss_;
    size_t block_size_;
    size_t key_size_;


    FixedReadWriter<Block> blocker_;
    FixedReadWriter<Key> keyer_;

    std::mutex mutex_;
};

template<typename Block, typename Key>
BlockStorage<Block,Key>::BlockStorage()
    : maximum_loaded_blocks_(0), index_size_(0), data_size_(0), file_size_(0),
      cache_hit_(0), cache_miss_(0), block_size_(0), key_size_(0)
{ }

template<typename Block, typename Key>
void BlockStorage<Block,Key>::close_file()
{
    block_file_.close();
}

template<typename Block, typename Key>
void BlockStorage<Block,Key>::open_file() 
{
    block_file_.open(path_, std::fstream::in | std::fstream::out | std::fstream::binary);
    if(!block_file_.is_open() && errno == ENOENT) {
        // create the file
        block_file_.open(path_, std::fstream::out | std::fstream::binary);
        block_file_.close();

        // reopen with in/out
        block_file_.open(path_, std::fstream::in | std::fstream::out | std::fstream::binary);
    }

    if(!block_file_.is_open()) {
        // get the error message from block_file_

        // create an error message with the filename using string streams
        std::stringstream ss;
        ss << "Could not open file: " << path_ << " - " << strerror(errno); 
        throw std::runtime_error(ss.str());
    }
    block_file_.seekg(0, std::ios::end);
    file_size_ = block_file_.tellg();
    block_file_.seekg(0, std::ios::beg);

    std::cerr << "file_size: " << file_size_ << std::endl;

    // if this is a new file, write out the data and index size (also zeros) to initialize the file
    if(file_size_ == 0) {
        block_file_.write(reinterpret_cast<const char *>(&data_size_), sizeof(data_size_));
        block_file_.write(reinterpret_cast<const char *>(&index_size_), sizeof(index_size_));
        block_file_.flush();
    }
}

template<typename Block, typename Key>
BlockStorage<Block,Key> BlockStorage<Block,Key>::create(string const & file_name, size_t maximum_loaded_blocks)
{
    BlockStorage<Block,Key> storage;
    storage.maximum_loaded_blocks_ = maximum_loaded_blocks;
    storage.path_ = file_name;
    
    storage.open_file();

    // figure out how big the block and key is
    auto pb = std::make_shared<Block>();
    auto pk = std::make_shared<Key>();

    auto temp_path = std::filesystem::temp_directory_path() / "blockstorage_test.blk";
    // open the temp_path fstream
    fstream t(temp_path, std::ios::out | std::ios::binary);

    // write it to the file and read the size
    auto b = t.tellp();
    storage.blocker_.write(t, *pb);
    auto e = t.tellp();

    storage.block_size_ = e - b;
    storage.keyer_.write(t, *pk);
    b = t.tellp();
    storage.key_size_ = b - e;

    std::cerr << "key size: " << storage.key_size_ << "\n";
    std::cerr << "block size: " << storage.block_size_ << std::endl;

    t.close();
    std::filesystem::remove(temp_path);

    return storage;
}

/* MUST execute under a lock */
template<typename Block, typename Key>
void BlockStorage<Block,Key>::remove_one() 
{
    if(keys_.size() == 0) return;

    auto key_to_remove = keys_.front();
    keys_.pop_front();
    loaded_.erase(key_to_remove);
}

/* must be executed under lock */
template<typename Block, typename Key>
void BlockStorage<Block,Key>::double_storage()
{
    size_t new_size = file_size_ * 2;

    close_file();
    std::filesystem::resize_file(path_, new_size);
    open_file();

    // copy the index from it's old location to it's new location
    auto buffer = std::make_unique<char[]>(index_size_);

    // move to the start of the index
    block_file_.seekg(-index_size_ - file_size_ - sizeof(index_size_) - sizeof(block_size_), std::ios::end);
    // read the index into the buffer
    block_file_.read(buffer, index_size_);
    // move to the start position of the new index
    block_file_.seekp(-index_size_- sizeof(index_size_) - sizeof(block_size_), std::ios::end);
    // write the new index
    block_file_.write(buffer, index_size_);
    block_file_.write(reinterpret_cast<const char *>(&block_size_), sizeof(block_size_));
    block_file_.write(reinterpret_cast<const char *>(&index_size_), sizeof(index_size_));

    // reset all our counters
    file_size_ = new_size;
}

// must be executed under lock
template<typename Block, typename Key>
size_t BlockStorage<Block,Key>::grow_index(Key const & key)
{
    if(data_size_ + index_size_ + key_size_ + sizeof(size_t) > file_size_) 
        double_storage();

    block_file_.seekp(-index_size_ - key_size_ - sizeof(size_t), std::ios::end);
    keyer_.write(key, block_file_);
    block_file_.write((char*)&next_index_, sizeof(size_t));
    auto ret = next_index_++;
    return ret;
}

template<typename Block, typename Key>
Block & BlockStorage<Block,Key>::get(Key const &key) 
{
    std::lock_guard<std::mutex> guard(mutex_);

    auto it = loaded_.find(key);
    if (it != loaded_.end()) 
    {
        ++cache_hit_;
        return *it->second;
    } else {
        ++cache_miss_;
        if (loaded_.size() >= maximum_loaded_blocks_) 
            remove_one(); 

        auto ip = index_.find(key);
        size_t dex = 0;
        bool init = false;
        if(ip != index_.end()) {
            dex = *ip;
        } else {
            dex = grow_index(key);
            init = true;
        }

        // load the block from disk
        auto block = std::make_shared<Block>();
        block_file_.seekg(dex * block_size_, std::ios::beg);
        if(init) {
            blocker_.write(block_file_, *block);
        } else {
            blocker_.read(block_file_, *block);
        }

        keys_.push_back(key);
        loaded_[key] = block;
        return *block;
    }
}

