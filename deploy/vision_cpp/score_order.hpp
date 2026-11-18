#pragma once
#include <algorithm>
#include <numeric>
#include <vector>
// Match NumPy 2.2's scalar argsort/quicksort (the deployed ARM build),
// including ties. Reverse the ascending index permutation only after sorting.
inline std::vector<int> score_order(const std::vector<float> &v) {
  std::vector<int> a(v.size());
  std::iota(a.begin(), a.end(), 0);
  if (a.empty())
    return a;
  struct Part {
    int l, h, d;
  };
  int depth = 0;
  for (size_t n = v.size(); n > 1; n >>= 1)
    ++depth;
  std::vector<Part> todo{{0, int(a.size()) - 1, 2 * depth}};
  auto less = [&](int i, int j) { return v[i] < v[j]; };
  while (!todo.empty()) {
    auto [l, h, d] = todo.back();
    todo.pop_back();
    if (d < 0) {
      // NumPy's 1-based max heap: equal children select the left child.
      int n = h - l + 1;
      auto sift = [&](int root, int end, int value) {
        while (root <= end / 2) {
          int child = root * 2;
          if (child < end && less(a[l + child - 1], a[l + child]))
            ++child;
          if (!less(value, a[l + child - 1]))
            break;
          a[l + root - 1] = a[l + child - 1];
          root = child;
        }
        a[l + root - 1] = value;
      };
      for (int i = n / 2; i >= 1; --i)
        sift(i, n, a[l + i - 1]);
      for (int end = n; end > 1; --end) {
        int value = a[l + end - 1];
        a[l + end - 1] = a[l];
        sift(1, end - 1, value);
      }
      continue;
    }
    while (h - l > 15) {
      int m = l + (h - l) / 2;
      if (less(a[m], a[l]))
        std::swap(a[m], a[l]);
      if (less(a[h], a[m]))
        std::swap(a[h], a[m]);
      if (less(a[m], a[l]))
        std::swap(a[m], a[l]);
      int pivot = a[m];
      std::swap(a[m], a[h - 1]);
      int i = l, j = h - 1;
      for (;;) {
        while (less(a[++i], pivot)) {
        }
        while (less(pivot, a[--j])) {
        }
        if (i >= j)
          break;
        std::swap(a[i], a[j]);
      }
      std::swap(a[i], a[h - 1]);
      --d;
      if (i - l < h - i) {
        todo.push_back({i + 1, h, d});
        h = i - 1;
      } else {
        todo.push_back({l, i - 1, d});
        l = i + 1;
      }
      if (d < 0) {
        todo.push_back({l, h, d});
        l = h;
        break;
      }
    }
    for (int i = l + 1; i <= h; ++i) {
      int value = a[i], j = i;
      while (j > l && less(value, a[j - 1])) {
        a[j] = a[j - 1];
        --j;
      }
      a[j] = value;
    }
  }
  std::reverse(a.begin(), a.end());
  return a;
}
