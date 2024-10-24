def reverse_by_n_elements(lst, n):
    result = []
    
    # Iterate over the list in steps of n
    for i in range(0, len(lst), n):
        group = lst[i:i + n]  # Get the next group of n elements
        
        # Reverse the current group manually
        reversed_group = []
        for j in range(len(group)-1, -1, -1):
            reversed_group.append(group[j])
        
        # Add the reversed group to the result
        result.extend(reversed_group)
    
    return result

# Example usage:
print(reverse_by_n_elements([1, 2, 3, 4, 5, 6, 7, 8], 3))  # Output: [3, 2, 1, 6, 5, 4, 8, 7]
print(reverse_by_n_elements([1, 2, 3, 4, 5], 2))           # Output: [2, 1, 4, 3, 5]
print(reverse_by_n_elements([10, 20, 30, 40, 50, 60, 70], 4)) # Output: [40, 30, 20, 10, 70, 60, 50]
