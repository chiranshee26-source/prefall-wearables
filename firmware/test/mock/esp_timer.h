#pragma once
#include <cstdint>
typedef void* esp_timer_handle_t;
struct esp_timer_create_args_t { void (*callback)(void*); void* arg; int dispatch_method; const char* name; bool skip_unhandled_events; };
inline int64_t esp_timer_get_time() { return 0; }
inline int esp_timer_create(const esp_timer_create_args_t*, esp_timer_handle_t*) { return 0; }
inline int esp_timer_stop(esp_timer_handle_t) { return 0; }
inline int esp_timer_start_once(esp_timer_handle_t, uint64_t) { return 0; }
