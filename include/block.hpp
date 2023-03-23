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
#include <thread>


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
    void save_one(Key const &key);
    // saves everything to disk
    void save_all();

    static BlockStorage<Block, Key> create_or_open(string const & file_name, size_t maximum_loaded_blocks);

    BlockStorage(BlockStorage<Block, Key> const &) = delete;
    BlockStorage(BlockStorage<Block, Key> &&) noexcept;
    ~BlockStorage();

    void dump(ostream & os);
private:
    BlockStorage();

    void remove_one_from_mem();
    void double_storage();
    std::shared_ptr<Block> grow_index(Key const & key);
    void open_file(size_t resize_old_size = 0);
    void close_file();
    void measure_sizes();

    void seekp_to_data();
    void seekp_to_block(size_t index);
    void seekp_to_index();
    void seekp_to_data_size();
    void seekp_to_index_size();
    void seekg_to_data();
    void seekg_to_block(size_t index);
    void seekg_to_index();
    void seekg_to_data_size();
    void seekg_to_index_size();

    bool read_block(Key const & key, Block & block);
    void update_file_size();
    void write_footer();

    std::map<Key,size_t> index_;
    size_t offset_;

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
    size_t footer_size_;

    FixedReadWriter<Block> blocker_;
    FixedReadWriter<Key> keyer_;

    std::mutex mutex_;
};

template<typename Block, typename Key>
BlockStorage<Block,Key>::BlockStorage()
    : offset_(0), maximum_loaded_blocks_(0), index_size_(0), data_size_(0), file_size_(0),
      cache_hit_(0), cache_miss_(0), block_size_(0), key_size_(0), footer_size_(sizeof(data_size_) + sizeof(index_size_))
{ }

template<typename Block, typename Key>
BlockStorage<Block,Key>::BlockStorage(BlockStorage<Block,Key> && other) noexcept
    : index_(std::move(other.index_)), offset_(other.offset_),
      loaded_(std::move(other.loaded_)), 
      keys_(std::move(other.keys_)), maximum_loaded_blocks_(other.maximum_loaded_blocks_), block_file_(std::move(other.block_file_)), path_(std::move(other.path_)),
      index_size_(other.index_size_), data_size_(other.data_size_), file_size_(other.file_size_),
      cache_hit_(other.cache_hit_), cache_miss_(other.cache_miss_), block_size_(other.block_size_), key_size_(other.key_size_), footer_size_(other.footer_size_),
      blocker_(std::move(other.blocker_)), keyer_(std::move(other.keyer_)),
      mutex_()
{ 
    std::cerr << "moved." << std::endl;
}


template<typename Block, typename Key>
BlockStorage<Block,Key>::~BlockStorage() 
{
    close_file();
}


// assumes under lock
template<typename Block, typename Key>
void BlockStorage<Block,Key>::update_file_size()
{
    block_file_.seekg(0, std::ios::end);
    file_size_ = block_file_.tellg();
    block_file_.seekg(0, std::ios::beg);
}

// assumes under lock
template<typename Block, typename Key>
bool BlockStorage<Block, Key>::read_block(Key const & key, Block & block) 
{
    auto it = index_.find(key);
    if(it == index_.end()) {
        return false;
    }

    block_file_.seekg(it->second * block_size_, std::ios::beg);
    blocker_.read(block_file_, block);
    return true;
}

template<typename Block, typename Key>
void BlockStorage<Block, Key>::dump(ostream & os) 
{
    // lock
    std::unique_lock<std::mutex> guard(mutex_);

    os << "BLOCK STORAGE: " << std::endl;
    os << "\tindex_size: " << index_size_ << std::endl;
    os << "\tdata_size: " << data_size_ << std::endl;
    os << "\tfile_size: " << file_size_ << std::endl;
    os << "\tcache_hit: " << cache_hit_ << std::endl;
    os << "\tcache_miss: " << cache_miss_ << std::endl;
    os << "\tblock_size: " << block_size_ << std::endl;
    os << "\tkey_size: " << key_size_ << std::endl;
    os << "\tmaximum_loaded_blocks: " << maximum_loaded_blocks_ << std::endl;
    os << "\tnext_index: " << offset_ << std::endl;
    os << "\tblocks loaded:\n";
    for(auto const & kv : loaded_) {
        os << kv.first << " ";
    }
    os << std::endl;
    os << "\tindex:\n";
    for(auto const & kv : index_) {
        os << "\t\t" << kv.first << ": " << kv.second << std::endl;
    }
    os << std::endl;
    os << "\tblocks:\n";
    auto pb = std::make_shared<Block>();
    for(auto const & kv : index_) {
        os << kv.first << ":\n";
        read_block(kv.first, *pb);
        os << *pb << std::endl;
    }
    os << std::endl;
    os << "done." << std::endl;
}

template<typename Block, typename Key>
void BlockStorage<Block,Key>::close_file()
{
    block_file_.close();
}

template<typename Block, typename Key>
void BlockStorage<Block,Key>::write_footer() 
{
    block_file_.write(reinterpret_cast<const char *>(&data_size_), sizeof(data_size_));
    block_file_.write(reinterpret_cast<const char *>(&index_size_), sizeof(index_size_));
    block_file_.flush();
}

template<typename Block, typename Key>
void BlockStorage<Block,Key>::measure_sizes()
{
    // figure out how big the block and key is
    auto pb = std::make_shared<Block>();
    auto pk = std::make_shared<Key>();

    auto temp_path = std::filesystem::temp_directory_path() / "blockstorage_test.blk";
    fstream t(temp_path, std::ios::out | std::ios::binary);

    // write it to the file and read the size
    auto b = t.tellp();

    // measure the size of a block written to disk
    blocker_.write(t, *pb);
    auto e = t.tellp();
    block_size_ = e - b;

    // measure the size of a key written to disk
    keyer_.write(t, *pk);
    b = t.tellp();
    key_size_ = b - e;

    std::cerr << "key size: " << key_size_ << "\n";
    std::cerr << "block size: " << block_size_ << std::endl;

    t.close();
    std::filesystem::remove(temp_path);
}

template<typename Block, typename Key>
void BlockStorage<Block,Key>::open_file(size_t resize_old_size) 
{
    measure_sizes();

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
    update_file_size();

    // if this is a new file, write out the data and index size (also zeros) to initialize the file
    if(file_size_ == 0) {
        std::cerr << "zero file size." << std::endl;
        write_footer();

        update_file_size();
    } else {
        if (resize_old_size > 0) {
            block_file_.seekg(resize_old_size - footer_size_, std::ios::beg);
            block_file_.read(reinterpret_cast<char*>(&data_size_), sizeof(data_size_));
            block_file_.read(reinterpret_cast<char*>(&index_size_), sizeof(index_size_));

            block_file_.seekg(resize_old_size - index_size_ - footer_size_, std::ios::beg);
            auto pk = std::make_shared<Key>();
            size_t off;
            for(size_t i = 0; i < index_size_; i += key_size_ + sizeof(size_t)) {
                keyer_.read(block_file_, *pk);
                block_file_.read(reinterpret_cast<char *>(&off), sizeof(size_t));

                index_[*pk] = off;
            }
        } else {
            // otherwise read the data and index size from the file
            std::cerr << "non-zero file size: " << file_size_ << std::endl;

            seekg_to_data_size();
            block_file_.read(reinterpret_cast<char *>(&data_size_), sizeof(data_size_));
            seekg_to_index_size();
            block_file_.read(reinterpret_cast<char *>(&index_size_), sizeof(index_size_));

            std::cerr << "data size: " << data_size_ << std::endl;
            std::cerr << "index size: " << index_size_ << std::endl;

            seekg_to_index();
            auto pk = std::make_shared<Key>();
            size_t off;

            for(size_t i = 0; i < index_size_; i += key_size_ + sizeof(size_t)) {
                keyer_.read(block_file_, *pk);
                block_file_.read(reinterpret_cast<char *>(&off), sizeof(size_t));

                index_[*pk] = off;
            }
        }
    }

    std::cerr << "file_size: " << file_size_ << std::endl;
}

template<typename Block, typename Key>
BlockStorage<Block,Key> BlockStorage<Block,Key>::create_or_open(string const & file_name, size_t maximum_loaded_blocks)
{
    BlockStorage<Block,Key> storage;
    storage.maximum_loaded_blocks_ = maximum_loaded_blocks;
    storage.path_ = file_name;

    storage.open_file();

    return storage;
}

template<typename Block, typename Key>
void BlockStorage<Block,Key>::save_one(Key const & key) 
{
    std::unique_lock<std::mutex> guard(mutex_);

    auto it = index_.find(key);
    if(it == index_.end()) 
        return;

    seekp_to_block(it->second);
    blocker_.write(block_file_, *loaded_[key]);
}

/* MUST execute under a lock */
template<typename Block, typename Key>
void BlockStorage<Block,Key>::remove_one_from_mem() 
{
    if(keys_.size() == 0) return;

    auto key_to_remove = keys_.front();
    std::cerr << "removing: " << key_to_remove << std::endl;
    keys_.pop_front();
    loaded_.erase(key_to_remove);
}

/* must be executed under lock */
template<typename Block, typename Key>
void BlockStorage<Block,Key>::double_storage()
{
    std::cerr << "double_storage()" << std::endl;

    size_t old_size = file_size_;
    size_t new_size = file_size_ * 2;

    std::cerr << "new_size: " << new_size << std::endl;

    close_file();
    std::filesystem::resize_file(path_, new_size);
    open_file(old_size);

    // copy the index from it's old location to it's new location
    auto buffer = std::make_unique<char[]>(index_size_);

    // move to the start of the index
    std::streamoff offset = 0;
    offset -= old_size + index_size_ + footer_size_;
    std::cerr << "seeking to index at: " << offset << std::endl;
    block_file_.seekg(offset, std::ios::end);
    // read the index into the buffer
    block_file_.read(buffer.get(), index_size_);
    // move to the start position of the new index

    offset = 0;
    offset -= index_size_ + footer_size_;
    std::cerr << "new index will be writtin at: " << offset << std::endl;
    block_file_.seekp(offset, std::ios::end);
    // write the new index
    block_file_.write(buffer.get(), index_size_);
    std::cerr << "block_file_.tellp(): " << block_file_.tellp() << std::endl;
    block_file_.write(reinterpret_cast<const char *>(&data_size_), sizeof(data_size_));
    std::cerr << "block_file_.tellp(): " << block_file_.tellp() << std::endl;
    block_file_.write(reinterpret_cast<const char *>(&index_size_), sizeof(index_size_));
    block_file_.flush();

    std::cerr << "END double_storage()" << std::endl;
}

// must be executed under lock
template<typename Block, typename Key>
std::shared_ptr<Block> BlockStorage<Block,Key>::grow_index(Key const & key)
{
    // do we have enough space for the new data?
    if( data_size_ + block_size_ +                  // current_data + new_data
        index_size_ + key_size_ + sizeof(size_t) +  // current_index + new_index
        footer_size_ >                              // footer
        file_size_) 
    {
        double_storage();
    }

    // move to the index area of our file
    seekp_to_index();
    block_file_.seekp(-key_size_ - sizeof(size_t), std::ios::cur); // move backward to fit the key and offset
    keyer_.write(block_file_, key);                                // write the key
    block_file_.write((char*)&offset_, sizeof(size_t));            // write the offset
    index_[key] = offset_;                                         // update the index in memory

    // write a blank block to the blocks
    auto pb = std::make_shared<Block>();
    seekp_to_block(offset_);
    blocker_.write(block_file_, *pb);

    // increment our counters
    data_size_ += block_size_;
    index_size_ += key_size_ + sizeof(size_t);

    // write the counters to disk
    seekp_to_data_size();
    // std::cerr << "block_file_.tellp(): " << block_file_.tellp() << std::endl;
    block_file_.write(reinterpret_cast<const char *>(&data_size_), sizeof(data_size_));
    seekp_to_index_size();
    // std::cerr << "block_file_.tellp(): " << block_file_.tellp() << std::endl;
    block_file_.write(reinterpret_cast<const char *>(&index_size_), sizeof(index_size_));
    block_file_.flush();

    // increment our offset
    offset_++;
    return pb;
}

template<typename Block, typename Key>
Block & BlockStorage<Block,Key>::get(Key const &key) 
{
    std::unique_lock<std::mutex> guard(mutex_);

    std::cerr << "get(" << key << ")" << std::endl;

    auto it = loaded_.find(key);
    if (it != loaded_.end()) 
    {
        std::cerr << "cache hit." << std::endl;
        ++cache_hit_;
        return *it->second;
    }

    std::cerr << "cache miss." << std::endl;
    ++cache_miss_;
    if (loaded_.size() >= maximum_loaded_blocks_) 
        remove_one_from_mem(); 

    std::shared_ptr<Block> block;

    auto ip = index_.find(key);
    if (ip == index_.end()) {
        std::cerr << "new key" << std::endl;
        block = grow_index(key);
    } else {    
        std::cerr << "old key, restore from disk: " << ip->second << std::endl;
        // load the block from disk
        block = std::make_shared<Block>();
        seekg_to_block(ip->second);
        blocker_.read(block_file_, *block);

        std::cerr << "value: " << *block << std::endl;
    }

    keys_.push_back(key);
    loaded_[key] = block;
    return *block;
}


template<typename Block, typename Key>
void BlockStorage<Block,Key>::seekp_to_data() 
{
    block_file_.seekp(0, std::ios::beg);
}
template<typename Block, typename Key>
void BlockStorage<Block,Key>::seekp_to_block(size_t index) 
{
    block_file_.seekp(index * block_size_, std::ios::beg);
}
template<typename Block, typename Key>
void BlockStorage<Block,Key>::seekp_to_index() 
{
    block_file_.seekp(-index_size_ - footer_size_, std::ios::end);
}
template<typename Block, typename Key>
void BlockStorage<Block,Key>::seekp_to_data_size() 
{
    block_file_.seekp(-footer_size_, std::ios::end);
}
template<typename Block, typename Key>
void BlockStorage<Block,Key>::seekp_to_index_size() 
{
    block_file_.seekp(-sizeof(index_size_), std::ios::end);
}
template<typename Block, typename Key>
void BlockStorage<Block,Key>::seekg_to_data() 
{
    block_file_.seekg(0, std::ios::beg);
}
template<typename Block, typename Key>
void BlockStorage<Block,Key>::seekg_to_block(size_t index) 
{
    block_file_.seekg(index * block_size_, std::ios::beg);
}
template<typename Block, typename Key>
void BlockStorage<Block,Key>::seekg_to_index() 
{
    block_file_.seekg(-index_size_ - footer_size_, std::ios::end);
}
template<typename Block, typename Key>
void BlockStorage<Block,Key>::seekg_to_data_size() 
{
    block_file_.seekg(-footer_size_, std::ios::end);
}
template<typename Block, typename Key>
void BlockStorage<Block,Key>::seekg_to_index_size() 
{
    block_file_.seekg(-sizeof(index_size_), std::ios::end);
}

