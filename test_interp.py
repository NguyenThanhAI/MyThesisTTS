import numpy as np

def interpolate_zeros(arr):
    # Tìm các vị trí không bằng 0
    non_zero_indices = np.where(arr != 0)[0]

    # Kiểm tra và xử lý nhóm các phần tử 0 ở đầu mảng
    if non_zero_indices[0] > 0:
        # Nội suy từ 0 đến phần tử khác 0 đầu tiên
        arr[:non_zero_indices[0]] = np.linspace(0, arr[non_zero_indices[0]], non_zero_indices[0] + 1)[:-1]

    # Kiểm tra và xử lý nhóm các phần tử 0 ở cuối mảng
    if non_zero_indices[-1] < len(arr) - 1:
        # Nội suy từ phần tử khác 0 cuối cùng đến 0
        num_zeros_at_end = len(arr) - non_zero_indices[-1] - 1
        arr[non_zero_indices[-1] + 1:] = np.linspace(arr[non_zero_indices[-1]], 0, num_zeros_at_end + 2)[1:]

    # Duyệt qua các giá trị không bằng 0 để nội suy các khoảng giữa
    for i in range(len(non_zero_indices) - 1):
        start, end = non_zero_indices[i], non_zero_indices[i + 1]
        
        # Nếu khoảng này có các giá trị 0 ở giữa
        if end - start > 1:
            arr[start + 1:end] = np.linspace(arr[start], arr[end], end - start + 1)[1:-1]
    
    return arr

# Ví dụ mảng 1
arr1 = np.array([0, 10, 30, 100])
result1 = interpolate_zeros(arr1)
print(result1)

# Ví dụ mảng 2
arr2 = np.array([0, 0, 0, 120, 200])
result2 = interpolate_zeros(arr2)
print(result2)

# Ví dụ mảng 3
arr3 = np.array([50, 100, 0, 0, 0])
result3 = interpolate_zeros(arr3)
print(result3)

# Ví dụ mảng 4
arr4 = np.array([10, 0, 0, 0, 120, 0, 0, 0])
result4 = interpolate_zeros(arr4)
print(result4)