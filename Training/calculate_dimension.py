def calc_dim(input_dim: int, kernel_size: int, stride: int, padding: int):

     return int((input_dim - kernel_size + 2 * padding)//stride + 1)