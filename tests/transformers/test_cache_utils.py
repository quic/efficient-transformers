# -----------------------------------------------------------------------------
#
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import pytest
import torch

from QEfficient.transformers.cache_utils import QEffDynamicCache  # Adjust the import based on your module's structure


@pytest.fixture
def setup_cache():
    return QEffDynamicCache()

def test_update_new_layer(setup_cache):
    cache = setup_cache
    key_states = torch.tensor([[[[1.0]]]])
    value_states = torch.tensor([[[[2.0]]]])
    layer_idx = 0

    k_out,v_out = cache.update(key_states, value_states, layer_idx)

    assert torch.equal(k_out, key_states)
    assert torch.equal(v_out, value_states)
    assert len(cache.key_cache) == 1
    assert len(cache.value_cache) == 1

def test_update_existing_layer(setup_cache):
    cache = setup_cache
    key_states = torch.tensor([[[[1.0]]]])
    value_states = torch.tensor([[[[2.0]]]])
    layer_idx = 0
    cache_kwargs = {
        'position_ids': torch.tensor([[0]]),
        'batch_index': torch.tensor([0])
    }
    cache.update(key_states, value_states, layer_idx,cache_kwargs)

    key_states2 = torch.tensor([[[[3.0]]]])
    value_states2 = torch.tensor([[[[4.0]]]])
    k_out,v_out = cache.update(key_states2, value_states2, layer_idx,cache_kwargs)

    assert torch.equal(k_out, key_states2)
    assert torch.equal(v_out, value_states2)
    assert len(cache.key_cache) == 1
    assert len(cache.value_cache) == 1

def test_update_with_batch_index(setup_cache):
    cache = setup_cache
    key_states = torch.tensor([[[[1.0]]]])
    value_states = torch.tensor([[[[2.0]]]])
    layer_idx = 0
    cache_kwargs = {
        'position_ids': torch.tensor([[0]])
    }
    cache.update(key_states, value_states, layer_idx,cache_kwargs)

    key_states2 = torch.tensor([[[[3.0]]]])
    value_states2 = torch.tensor([[[[4.0]]]])
    k_out,v_out = cache.update(key_states2, value_states2, layer_idx,cache_kwargs)

    assert torch.equal(k_out, key_states2)
    assert torch.equal(v_out, value_states2)
    assert len(cache.key_cache) == 1
    assert len(cache.value_cache) == 1

def test_update3D_new_layer(setup_cache):
    cache = setup_cache
    key_states = torch.tensor([[[[1.0]]]])
    value_states = torch.tensor([[[[2.0]]]])
    layer_idx = 0

    k_out,v_out = cache.update3D(key_states, value_states, layer_idx)

    assert torch.equal(k_out, key_states)
    assert torch.equal(v_out, value_states)
    assert len(cache.key_cache) == 1
    assert len(cache.value_cache) == 1

def test_update3D_existing_layer(setup_cache):
    cache = setup_cache
    key_states = torch.tensor([[[[1.0]]]])
    value_states = torch.tensor([[[[2.0]]]])
    layer_idx = 0
    cache_kwargs = {
        'position_ids': torch.tensor([[0]]),
        'batch_index': torch.tensor([0])
    }
    cache.update3D(key_states, value_states, layer_idx,cache_kwargs)

    key_states2 = torch.tensor([[[[3.0]]]])
    value_states2 = torch.tensor([[[[4.0]]]])
    k_out,v_out = cache.update3D(key_states2, value_states2, layer_idx,cache_kwargs)

    assert torch.equal(k_out, key_states2)
    assert torch.equal(v_out, value_states2)
    assert len(cache.key_cache) == 1
    assert len(cache.value_cache) == 1

def test_update3D_with_batch_index(setup_cache):
    cache = setup_cache
    key_states = torch.tensor([[[[1.0]]]])
    value_states = torch.tensor([[[[2.0]]]])
    layer_idx = 0
    cache_kwargs = {
        'position_ids': torch.tensor([[0]])
    }
    cache.update3D(key_states, value_states, layer_idx,cache_kwargs)

    key_states2 = torch.tensor([[[[3.0]]]])
    value_states2 = torch.tensor([[[[4.0]]]])
    k_out,v_out = cache.update3D(key_states2, value_states2, layer_idx,cache_kwargs)

    assert torch.equal(k_out, key_states2)
    assert torch.equal(v_out, value_states2)
    assert len(cache.key_cache) == 1
    assert len(cache.value_cache) == 1
