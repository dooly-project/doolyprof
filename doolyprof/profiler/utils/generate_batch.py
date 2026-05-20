import random
from itertools import product
from typing import List, Union


class AttentionBatchConfig:
    """
    Configuration for an attention operation batch, following Vidur's structure.

    This preserves all necessary information for proper KV cache initialization:
    - For prefill operations: how much is new vs existing KV cache
    - For decode operations: the existing KV cache size per request

    Attributes:
        prefill_chunk_size: Number of new tokens being processed (0 for decode)
        kv_cache_size: Size of existing KV cache per request
        batch_size: Number of requests in the batch
        is_prefill: Whether this is a prefill (True) or decode (False) operation
    """
    def __init__(
        self,
        prefill_chunk_size: int,
        kv_cache_size: int,
        batch_size: int,
        is_prefill: bool,
    ):
        self.prefill_chunk_size = prefill_chunk_size
        self.kv_cache_size = kv_cache_size
        self.batch_size = batch_size
        self.is_prefill = is_prefill

    def is_valid(self, max_seq_len: int) -> bool:
        """Validate the configuration following Vidur's validation logic."""
        if self.is_prefill:
            if self.batch_size != 1:
                return False
            elif self.prefill_chunk_size == 0:
                return False
            elif self.prefill_chunk_size + self.kv_cache_size > max_seq_len:
                return False
        else:  # decode
            if self.prefill_chunk_size > 0:
                return False
            elif self.kv_cache_size == 0:
                return False
            elif self.kv_cache_size > max_seq_len:
                return False
        return True

    def is_under_memory_limit(self, max_num_tokens: int) -> bool:
        """
        Check if this configuration fits within memory limits.

        Args:
            max_num_tokens: Maximum number of tokens that can fit in KV cache

        Returns:
            True if the configuration fits, False otherwise
        """
        return (
            self.batch_size * (self.kv_cache_size + self.prefill_chunk_size)
            <= max_num_tokens
        )

    def get_kv_cache_list(self) -> List[int]:
        """
        Convert to list format compatible with existing code.
        Returns a list of KV cache sizes (one per request in batch).
        """
        if self.is_prefill:
            # For prefill: total tokens = prefill_chunk + existing_kv
            total_tokens = self.prefill_chunk_size + self.kv_cache_size
            return [total_tokens] * self.batch_size
        else:
            # For decode: each request has kv_cache_size tokens
            return [self.kv_cache_size] * self.batch_size

    @property
    def num_tokens(self) -> int:
        if self.is_prefill:
            return self.batch_size * self.prefill_chunk_size
        else:
            return self.batch_size

    def __str__(self):
        return (f"AttentionBatchConfig(prefill_chunk={self.prefill_chunk_size}, "
                f"kv_cache={self.kv_cache_size}, batch={self.batch_size}, "
                f"is_prefill={self.is_prefill})")

    def __repr__(self):
        return self.__str__()


def get_attention_batch_sizes_to_profile(min_batch_size: int, max_batch_size: int):
    """
    Get batch sizes to profile following Vidur's strategy.

    Args:
        min_batch_size: Minimum batch size to include
        max_batch_size: Maximum batch size to include

    Returns:
        List of batch sizes to profile
    """
    BATCH_SIZE_SPACE = list(range(1, 128 + 1, 1)) + list(range(128, 1024 + 1, 8))
    return list(
        filter(
            lambda x: (x >= min_batch_size and x <= max_batch_size), BATCH_SIZE_SPACE
        )
    )


def get_attention_prefill_chunk_sizes_to_profile(max_seq_len: int):
    PREFILL_CHUNK_SIZE_SPACE = (
        list(range(4, 128 + 1, 8))
        + list(range(128, 1024 + 1, 32))
        + list(range(1024, 4 * 1024 + 1, 64))
        + list(range(4 * 1024, 16 * 1024 + 1, 128))
        + list(range(16 * 1024, 64 * 1024 + 1, 256))
    )
    prefill_chunk_sizes_to_profile = []
    for prefill_chunk_size in PREFILL_CHUNK_SIZE_SPACE:
        if prefill_chunk_size <= max_seq_len:
            prefill_chunk_sizes_to_profile.append(prefill_chunk_size)
        else:
            break
    return prefill_chunk_sizes_to_profile


def get_seq_lengths_to_profile(max_seq_len: int):
    """
    Get sequence lengths to profile following Vidur's strategy.

    Args:
        max_seq_len: Maximum sequence length

    Returns:
        List of sequence lengths to profile
    """
    SEQ_LENGTH_SIZE_SPACE = (
        list(range(0, 1024 + 1, 32))
        + list(range(1024, 4 * 1024 + 1, 64))
        + list(range(4 * 1024, 64 * 1024 + 1, 256))
    )
    seq_lengths_to_profile = []
    for seq_length in SEQ_LENGTH_SIZE_SPACE:
        if seq_length < max_seq_len:
            seq_lengths_to_profile.append(seq_length)
        else:
            break
    return seq_lengths_to_profile


def distribute_dust(batch, total_kv_cache_size, max_value, start_idx=1):
    """
    Distribute remaining tokens to maintain exact sum while preserving max_value constraint.

    Args:
        batch: List of token counts to adjust
        total_kv_cache_size: Target sum that batch should equal
        max_value: Maximum value that any element in batch can have
        start_idx: Starting index for distributing tokens (default 1 to preserve batch[0])

    Returns:
        Adjusted batch list
    """
    diff = total_kv_cache_size - sum(batch)

    if diff > 0:
        # Add tokens to elements starting from start_idx without exceeding max_value
        for i in range(start_idx, len(batch)):
            if diff == 0:
                break
            if batch[i] + 1 <= max_value:
                batch[i] += 1
                diff -= 1

        # If still have leftover, cycle through again
        while diff > 0:
            added_any = False
            for i in range(start_idx, len(batch)):
                if diff == 0:
                    break
                if batch[i] + 1 <= max_value:
                    batch[i] += 1
                    diff -= 1
                    added_any = True

            # If we couldn't add any tokens without exceeding max, handle edge case
            if not added_any:
                # For gradient mode, add to first element if we can't distribute elsewhere
                if start_idx > 0 and diff > 0:
                    batch[0] += diff
                    diff = 0
                break

    elif diff < 0:
        # Remove tokens from elements starting from the end
        diff = abs(diff)
        for i in range(len(batch) - 1, start_idx - 1, -1):
            if diff == 0:
                break
            if batch[i] > 1:
                batch[i] -= 1
                diff -= 1

    return batch


def generate_gradient_inputs(batch_size, total_kv_cache_size, test_counts):
    """
    Generate gradient mode inputs where requests have smoothly decreasing sizes.
    Maximum request size stays constant; only the slope/gradient varies.

    Args:
        batch_size: Number of requests in batch
        total_kv_cache_size: Total tokens across all requests
        test_counts: Number of different configurations to generate

    Returns:
        List of batch configurations
    """
    uniform_size = total_kv_cache_size // batch_size
    kv_cache_sizes = [[uniform_size] * batch_size]

    step = (total_kv_cache_size - uniform_size) // (test_counts - 1)
    max_kv_cache_size_list = [uniform_size + i * step for i in range(1, test_counts)]

    for max_size in max_kv_cache_size_list:
        # Binary Search to find the correct slope
        # We want: sum([max(1, max_size - slope * i)]) == total_kv_cache_size

        low_slope = 0.0
        high_slope = float(max_size)  # Max possible slope
        best_slope = 0.0

        # 20-30 iterations is plenty for float precision
        for _ in range(30):
            mid_slope = (low_slope + high_slope) / 2
            current_sum = sum(max(1, int(max_size - mid_slope * i)) for i in range(batch_size))

            if current_sum < total_kv_cache_size:
                # Sum is too small -> Slope is too steep -> Decrease slope
                high_slope = mid_slope
            else:
                # Sum is too large -> Slope is too shallow -> Increase slope
                low_slope = mid_slope

        # Use the high_slope (slightly steeper) to ensure we are UNDER the budget
        # so we can fill up the dust, rather than remove from Max.
        best_slope = high_slope

        batch = [max(1, int(max_size - best_slope * i)) for i in range(batch_size)]

        # Distribute remaining 'dust' to indices 1..N to preserve the max at index 0
        batch = distribute_dust(batch, total_kv_cache_size, batch[0], start_idx=1)

        kv_cache_sizes.append(batch)

    return kv_cache_sizes


def generate_single_max_inputs(batch_size, total_kv_cache_size, test_counts):
    """
    Generate single-max mode inputs where one request is large and others are uniform.
    Maximum request size stays constant across the batch.

    Args:
        batch_size: Number of requests in batch
        total_kv_cache_size: Total tokens across all requests
        test_counts: Number of different configurations to generate

    Returns:
        List of batch configurations
    """
    uniform_size = total_kv_cache_size // batch_size
    kv_cache_sizes = [[uniform_size] * batch_size]

    step = (total_kv_cache_size - uniform_size) // (test_counts - 1)
    max_kv_cache_size_list = [uniform_size + i * step for i in range(1, test_counts)]

    for max_size in max_kv_cache_size_list:
        # Calculate size for other (non-max) requests
        remaining = total_kv_cache_size - max_size
        other_size = remaining // (batch_size - 1)

        if other_size < 1:
            other_size = 1

        # Create batch: [max_size, other_size, other_size, ...]
        batch = [max_size] + ([other_size] * (batch_size - 1))

        # Distribute remainder to maintain exact sum without exceeding max_size
        batch = distribute_dust(batch, total_kv_cache_size, max_size, start_idx=1)

        kv_cache_sizes.append(batch)

    return kv_cache_sizes


def generate_multi_max_inputs(batch_size, total_kv_cache_size, test_counts):
    """
    Generate multi-max mode inputs where multiple requests can have max size.
    Maximum request size stays constant; the number of max-sized requests varies.

    Args:
        batch_size: Number of requests in batch
        total_kv_cache_size: Total tokens across all requests
        test_counts: Number of different configurations to generate

    Returns:
        List of batch configurations
    """
    uniform_size = total_kv_cache_size // batch_size
    kv_cache_sizes = [[uniform_size] * batch_size]

    step = (total_kv_cache_size - uniform_size) // (test_counts - 1)
    max_kv_cache_size_list = [uniform_size + i * step for i in range(1, test_counts)]

    for new_max_size in max_kv_cache_size_list:
        # Calculate how many requests can be at max size
        max_request_count = total_kv_cache_size // new_max_size

        if max_request_count >= batch_size:
            # All requests can be at max size
            max_request_count = batch_size
            other_size = new_max_size
        elif max_request_count == 0:
            # Max size too large for even one request
            continue
        else:
            # Some requests at max, others smaller
            remaining_space = total_kv_cache_size - (new_max_size * max_request_count)
            remaining_requests = batch_size - max_request_count
            other_size = remaining_space // remaining_requests

            if other_size < 1:
                # Adjust to ensure other_size is at least 1
                max_request_count = max(1, (total_kv_cache_size - remaining_requests) // new_max_size)
                remaining_space = total_kv_cache_size - (new_max_size * max_request_count)
                remaining_requests = batch_size - max_request_count
                other_size = max(1, remaining_space // remaining_requests)

        # Create batch: [max, max, ..., other, other, ...]
        batch = [new_max_size] * max_request_count + [other_size] * (batch_size - max_request_count)

        # Distribute remainder to non-max elements without exceeding new_max_size
        batch = distribute_dust(batch, total_kv_cache_size, new_max_size, start_idx=max_request_count)

        # Only add valid batches (all elements >= 1)
        if min(batch) >= 1:
            kv_cache_sizes.append(batch)

    return kv_cache_sizes


def generate_random_core_tests(
    batch_size_range=(2, 64),
    total_kv_range=(256, 32768),
    num_per_mode=1,
    seed=None,
    test_count_range=(2, 8),
):
    """
    Generate randomized scenarios (uniform, single-max, gradient, multi-max) with
    random batch size and total KV length. Returns num_per_mode configs per mode.

    Args:
        batch_size_range: (min, max) for random batch size (inclusive).
        total_kv_range: (min, max) for random total KV tokens (inclusive).
        num_per_mode: Number of configs to sample for each mode.
        seed: Optional seed for reproducibility.
        test_count_range: (min, max) number of configurations to generate per mode before sampling one.

    Returns:
        List of batch configurations: num_per_mode * 4 total.
    """
    rng = random.Random(seed)

    def sample_one(mode_fn):
        bs = rng.randint(*batch_size_range)
        total_kv = rng.randint(*total_kv_range)
        tcount = rng.randint(*test_count_range)

        if mode_fn is None:
            cfg = [total_kv // bs] * bs
            return distribute_dust(cfg, total_kv, max_value=total_kv, start_idx=0)

        configs = mode_fn(bs, total_kv, test_counts=tcount)
        return configs[rng.randrange(len(configs))] if configs else [total_kv // bs] * bs

    batches = []
    for _ in range(num_per_mode):
        batches.append(sample_one(None))  # uniform
    for _ in range(num_per_mode):
        batches.append(sample_one(generate_single_max_inputs))
    for _ in range(num_per_mode):
        batches.append(sample_one(generate_gradient_inputs))
    for _ in range(num_per_mode):
        batches.append(sample_one(generate_multi_max_inputs))

    return batches


def generate_vidur_num_tokens(max_size: int) -> List[int]:
    """
    Generate token count inputs following Vidur's MLP profiling strategy.

    For non-attention operations (MLP, linear, embedding, etc.), we only need
    total token counts, not batch structure.

    Args:
        max_tokens: Maximum number of tokens to profile

    Returns:
        List of single-element lists representing total token counts
        (wrapped in lists for compatibility with batch interface)
    """
    NUM_TOKENS_SPACE = (
        list([1, 2, 4])
        + list(range(8, 1024, 8))
        + list(range(1024, 2 * 1024 + 1, 16))
        + list(range(2 * 1024, 4 * 1024 + 1, 32))
        + list(range(4 * 1024, 8 * 1024 + 1, 64))
        + list(range(8 * 1024, 16 * 1024 + 1, 128))
        + list(range(16 * 1024, 32 * 1024 + 1, 256))
        + list(range(32 * 1024, 64 * 1024 + 1, 512))
        + list(range(64 * 1024, 128 * 1024 + 1, 1024))
    )

    num_tokens_to_profile = []
    for num_tokens in NUM_TOKENS_SPACE:
        if num_tokens <= max_size:
            num_tokens_to_profile.append(num_tokens)
        else:
            break

    return num_tokens_to_profile


def generate_vidur_inputs(
    max_seq_len: int,
    min_batch_size: int = 1,
    max_batch_size: int = 128,
    profile_only_prefill: bool = False,
    profile_only_decode: bool = False,
) -> List[AttentionBatchConfig]:
    input_combinations = []

    # 1. Chunked Prefills
    # Format: AttentionBatchConfig(prefill_chunk_size, kv_cache_size, 1, True)
    if not profile_only_decode:
        prefill_chunk_sizes = get_attention_prefill_chunk_sizes_to_profile(max_seq_len)
        for prefill_chunk_size in prefill_chunk_sizes:
            num_partitions = max_seq_len // prefill_chunk_size
            kv_cache_sizes = [
                partition_index * prefill_chunk_size
                for partition_index in range(num_partitions)
            ]
            for kv_cache_size in kv_cache_sizes:
                config = AttentionBatchConfig(
                    prefill_chunk_size=prefill_chunk_size,
                    kv_cache_size=kv_cache_size,
                    batch_size=1,
                    is_prefill=True
                )
                if config.is_valid(max_seq_len):
                    input_combinations.append(config)

    # 2. Full Prefills
    # Format: AttentionBatchConfig(prefill_length, 0, 1, True)
    if not profile_only_decode:
        prefill_lengths = get_seq_lengths_to_profile(max_seq_len)
        for prefill_length in prefill_lengths:
            if prefill_length > 0:  # Skip zero-length prefills
                config = AttentionBatchConfig(
                    prefill_chunk_size=prefill_length,
                    kv_cache_size=0,
                    batch_size=1,
                    is_prefill=True
                )
                if config.is_valid(max_seq_len):
                    input_combinations.append(config)

    # 3. Decodes
    # Format: AttentionBatchConfig(0, kv_cache_size, batch_size, False)
    if not profile_only_prefill:
        kv_cache_sizes = get_seq_lengths_to_profile(max_seq_len)
        batch_sizes = get_attention_batch_sizes_to_profile(min_batch_size, max_batch_size)

        for kv_cache_size, batch_size in product(kv_cache_sizes, batch_sizes):
            config = AttentionBatchConfig(
                prefill_chunk_size=0,
                kv_cache_size=kv_cache_size,
                batch_size=batch_size,
                is_prefill=False,
                # is_prefill=True, # force prefill kernel usage (vllm does this too)
            )
            if config.is_valid(max_seq_len):
                input_combinations.append(config)

    return input_combinations
         
