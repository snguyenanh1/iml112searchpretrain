# Retry Logic Implementation for iML Manager

## Tóm tắt

Đã thêm **retry logic** toàn diện cho tất cả các agent calls trong iML Manager để xử lý lỗi API (đặc biệt là 503 Service Unavailable errors).

## Vấn đề đã giải quyết

### Trước đây:
- Retry logic CHỈ tồn tại trong `base_chat.py` (LLM layer)
- Nếu lỗi xảy ra ở manager layer (trước khi đến LLM call), retry logic KHÔNG được trigger
- Khi gặp 503 errors ở API level, pipeline sẽ thất bại ngay lập tức

### Hiện tại:
- ✅ Retry logic được áp dụng cho TẤT CẢ agent calls trong manager
- ✅ Xử lý đặc biệt cho 503 errors: chờ 5 phút, retry tối đa 10 lần
- ✅ Exponential backoff cho các lỗi khác: 32s, 64s, 128s...
- ✅ Logging rõ ràng với emoji: 🔄, ⏳, 🔁, ❌

## Các thay đổi chính

### 1. Thêm retry decorator và helper function
**File:** `src/iML/core/manager.py`

```python
def is_503_error(exception):
    """Kiểm tra xem exception có phải là 503 Service Unavailable error không"""
    # Kiểm tra nhiều patterns: HTTP 503, gRPC UNAVAILABLE, etc.

def retry_on_api_error(max_retries: int = 10):
    """Decorator để retry agent calls với intelligent backoff"""
    # 503 errors: chờ 5 phút
    # Các lỗi khác: exponential backoff (32s, 64s, 128s...)
```

### 2. Thêm wrapper methods cho tất cả agents
Mỗi agent có một wrapper method với retry decorator:

```python
@retry_on_api_error(max_retries=10)
def _call_description_analyzer(self):
    return self.description_analyzer_agent()

@retry_on_api_error(max_retries=10)
def _call_profiling_agent(self):
    return self.profiling_agent()

@retry_on_api_error(max_retries=10)
def _call_profiling_summarizer(self):
    return self.profiling_summarizer_agent()

@retry_on_api_error(max_retries=10)
def _call_model_retriever(self):
    return self.model_retriever_agent()

@retry_on_api_error(max_retries=10)
def _call_guideline_agent(self, iteration_type=None):
    if iteration_type:
        return self.guideline_agent(iteration_type=iteration_type)
    return self.guideline_agent()

@retry_on_api_error(max_retries=10)
def _call_preprocessing_coder(self, iteration_type=None):
    if iteration_type:
        return self.preprocessing_coder_agent(iteration_type=iteration_type)
    return self.preprocessing_coder_agent()

@retry_on_api_error(max_retries=10)
def _call_modeling_coder(self, iteration_type=None):
    if iteration_type:
        return self.modeling_coder_agent(iteration_type=iteration_type)
    return self.modeling_coder_agent()

@retry_on_api_error(max_retries=10)
def _call_assembler(self, iteration_type=None):
    if iteration_type:
        return self.assembler_agent(iteration_type=iteration_type)
    return self.assembler_agent()

@retry_on_api_error(max_retries=10)
def _call_comparison_agent(self, iteration_results, original_task_description):
    return self.comparison_agent(
        iteration_results=iteration_results,
        original_task_description=original_task_description
    )

@retry_on_api_error(max_retries=10)
def _call_ensemble_agent(self, iteration_paths, iteration_results):
    return self.ensemble_agent(
        iteration_paths=iteration_paths,
        iteration_results=iteration_results
    )
```

### 3. Cập nhật tất cả pipeline methods
Đã cập nhật các methods sau để sử dụng wrapper methods với retry:

- ✅ `run_pipeline()`
- ✅ `run_pipeline_multi_iteration()`
- ✅ `run_pipeline_partial()`
- ✅ `resume_pipeline_from_checkpoint()`
- ✅ `_run_shared_analysis()`
- ✅ `_run_iteration_pipeline()`
- ✅ `_run_iteration_pipeline_with_checks()`

## Retry Logic Details

### Đối với 503 Service Unavailable errors:
```
Attempt 1 fails (503) 
→ 🔄 Log warning
→ ⏳ Wait 5 minutes (300s)
→ 🔁 Retry attempt 2
... (lặp lại tối đa 10 lần)
→ ❌ Sau 10 lần: raise exception
```

### Đối với các errors khác:
```
Attempt 1 fails 
→ ⚠️ Log error
→ ⏳ Wait 32 seconds
→ 🔁 Retry attempt 2

Attempt 2 fails
→ ⏳ Wait 64 seconds
→ 🔁 Retry attempt 3

Attempt 3 fails
→ ⏳ Wait 128 seconds
→ 🔁 Retry attempt 4

... (max 128s, tối đa 10 lần)
→ ❌ Sau 10 lần: raise exception
```

## Ví dụ Log Output

### Khi gặp 503 error:
```
🔄 Attempt 1/10 - Service Unavailable (503). API is overloaded.
⏳ Waiting 300 seconds (5 minutes) before retry...
🔁 Retrying agent call (attempt 2/10)...
```

### Khi gặp error khác:
```
⚠️  Attempt 1/10 failed: ConnectionError: API connection timeout
⏳ Waiting 32 seconds before retry...
🔁 Retrying agent call (attempt 2/10)...
```

### Khi hết retries:
```
❌ Max retries (10) reached for 503 errors. Giving up.
```

## Lợi ích

1. **Tăng độ ổn định**: Pipeline không bị fail ngay lập tức khi gặp lỗi tạm thời
2. **Xử lý 503 errors tốt hơn**: Chờ đủ lâu để API phục hồi (5 phút)
3. **Tiết kiệm thời gian**: Tự động retry thay vì phải chạy lại toàn bộ pipeline
4. **Logging rõ ràng**: Dễ dàng theo dõi retry attempts và debug
5. **Không phá vỡ code cũ**: Tất cả agent calls vẫn hoạt động như trước, chỉ thêm retry layer

## Testing

Để test retry logic, bạn có thể:

1. **Tắt internet tạm thời** trong khi pipeline đang chạy
2. **Sử dụng API key hết quota** để trigger 503 errors
3. **Kiểm tra logs** để xác nhận retry attempts được thực hiện

## Notes

- Retry logic chỉ áp dụng cho **agent calls**, không áp dụng cho code execution
- Timeout của iteration vẫn được giữ nguyên và ưu tiên cao hơn retry logic
- Nếu timeout xảy ra trong khi đang retry, pipeline sẽ dừng ngay lập tức

