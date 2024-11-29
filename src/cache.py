# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed according to the terms of the
# GNU General Public License version 2.

import math, block, response
import pprint
from replacement_policy import * 

class Cache:
    def __init__(self, name, word_size, block_size, n_blocks, associativity, hit_time, write_time, write_back, logger, next_level=None, rep_policy='', verbose=False):
        #Parameters configured by the user
        self.name = name
        self.word_size = word_size
        self.block_size = block_size
        self.n_blocks = n_blocks
        self.associativity = associativity
        self.hit_time = hit_time
        self.clflush_time = hit_time # assume flush is as fast as hit since
        self.write_time = write_time
        self.write_back = write_back
        self.logger = logger
        self.logger.disabled = False#True
        self.set_rep_policy = {}
        self.verbose = verbose
        if rep_policy == 'lru':
            self.vprint("use lru") 
            self.rep_policy = lru_policy
        else:
            self.rep_policy = lru_policy
            if name == 'cache_1':
                self.vprint("no rep_policy specified or policy specified not exist")
                self.vprint("use lru_policy")

        #Total number of sets in the cache
        self.n_sets =int( n_blocks / associativity )
        
        #Dictionary that holds the actual cache data
        self.data = {}
        self.set = {}

        #Pointer to the next lowest level of memory
        #Main memory gets the default None value
        self.next_level = next_level

        #Figure out spans to cut the binary addresses into block_offset, index, and tag
        self.block_offset_size = int(math.log(self.block_size, 2))
        self.index_size = int(math.log(self.n_sets, 2))

        #Initialize the data dictionary
        if next_level:
            for i in range(self.n_sets):
                index = str(bin(i))[2:].zfill(self.index_size)
                if index == '':
                    index = '0'
                self.data[index] = []    # use array of blocks for each set
                for j in range(associativity):
                    # isntantiate with empty tags
                    self.data[index].append((INVALID_TAG, block.Block(self.block_size, 0, False, 'x')))

                self.set_rep_policy[index] = self.rep_policy(associativity, block_size) 

    def vprint(self, *args):
        if self.verbose == 1:
            print( " "+" ".join(map(str,args))+" ")

    # flush the cache line that contains the address from all cache hierachy
    # since flush is does not affect memory domain_id
    def clflush(self, address, current_step):
        #address = address.zfill(8)
        #print(address) 
        r = response.Response({self.name:True}, self.clflush_time) #flush regardless 
        #Parse our address to look through this cache
        block_offset, index, tag = self._parse_address(address)
        #Get the tags in this set
        in_cache = []
        for i in range( 0, len(self.data[index]) ):
            if self.data[index][i][0] != INVALID_TAG:#'x':
                in_cache.append(self.data[index][i][0])

        #If this tag exists in the set, this is a hit
        if tag in in_cache:
            #print(tag + ' in cache')
            for i in range( 0, len(self.data[index])):
                if self.data[index][i][0] == tag: 
                    #print(self.data[index][i][1].address)
                    self.data[index][i] = (INVALID_TAG, block.Block(self.block_size, current_step, False, ''))
                    break
            self.set_rep_policy[index].invalidate(tag)

        # clflush from the next level of memory
        if self.next_level != None and self.next_level.name != "mem":
            self.next_level.clflush(address, current_step)
        return r 

    def read(self, address, current_step):
        r = None
        #Check if this is main memory
        #Main memory is always a hit
        if not self.next_level:
            r = response.Response({self.name:True}, self.hit_time)
            evict_addr = -1
        else:
            #Parse our address to look through this cache
            block_offset, index, tag = self._parse_address(address)
            
            #Get the tags in this set
            in_cache = []
            for i in range( 0, len(self.data[index]) ):
                if self.data[index][i][0] != INVALID_TAG:#'x':
                    in_cache.append(self.data[index][i][0])

            #If this tag exists in the set, this is a hit
            if tag in in_cache:
                #print(tag + 'in cache')
                for i in range( 0, len(self.data[index])):
                    if self.data[index][i][0] == tag: 
                        self.data[index][i][1].read(current_step)
                        break
                self.set_rep_policy[index].touch(tag, current_step)
                
                r = response.Response({self.name:True}, self.hit_time)
                evict_addr = -1 #no evition needed
            else:
                #Read from the next level of memory
                r, evict_addr = self.next_level.read(address, current_step )
                r.deepen(self.write_time, self.name)

                in_cache = []
                for i in range( 0, len(self.data[index]) ):
                    if self.data[index][i][0] != INVALID_TAG:#'x':
                        in_cache.append(self.data[index][i][0])
                
                #If there's space in this set, add this block to it
                if len(in_cache) < self.associativity:
                    #print('a')
                    for i in range( 0, len(self.data[index])):
                        if self.data[index][i][0] == INVALID_TAG:#'x':
                            self.data[index][i] = (tag, block.Block(self.block_size, current_step, False, address))
                            break
                    self.set_rep_policy[index].instantiate_entry(tag, current_step)
                    
                else:
                    #Find the victim block and replace it
                    victim_tag = self.set_rep_policy[index].find_victim(current_step)
                    if victim_tag != INVALID_TAG: 
                        # Write the block back down if it's dirty and we're using write back
                        if self.write_back:
                            for i in range( 0, len(self.data[index])):
                                if self.data[index][i][0] == victim_tag:
                                    if self.data[index][i][1].is_dirty():  
                                        self.logger.info('\tWriting back block ' + address + ' to ' + self.next_level.name)
                                        temp, _, _ = self.next_level.write(self.data[index][i][1].address, True, current_step)
                                        r.time += temp.time
                                        break
                        # Delete the old block and write the new one
                        for i in range( 0, len(self.data[index])):
                            if self.data[index][i][0] == victim_tag:
                                self.data[index][i] = (tag, block.Block(self.block_size, current_step, False, address))
                                break    
                        if int(self.n_blocks/ self.associativity) == 1:
                            indexi = ''
                        else:
                            indexi = index
                        evict_addr = victim_tag  + indexi  + '0' *  int(math.log(self.block_size,2))# assume line size is always 1B for different level
                        self.set_rep_policy[index].invalidate(victim_tag)
                        self.set_rep_policy[index].instantiate_entry(tag, current_step)
                    else:
                        evict_addr = -1
        return r, evict_addr

    def write(self, address, from_cpu, current_step ):
        return NotImplementedError('write not implemented')

    def _parse_address(self, address):
        #Calculate our address length and convert the address to binary string
        address_size = len(address) * 4
        binary_address = bin(int(address, 16))[2:].zfill(address_size)

        if self.block_offset_size > 0:
            block_offset = binary_address[-self.block_offset_size:]
            index = binary_address[-(self.block_offset_size+self.index_size):-self.block_offset_size]
            if index == '':
                index = '0'
            tag = binary_address[:-(self.block_offset_size+self.index_size)]
        else:
            block_offset = '0'
            if self.index_size != 0:
                index = binary_address[-(self.index_size):]
                tag = binary_address[:-self.index_size] 
            else:
                index = '0'
                tag = binary_address

        return (block_offset, index, tag)

class InvalidOpError(Exception):
    pass
