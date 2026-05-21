struct ComplexField {
    cuComplex* data;        // GPU память, [rows × cols]
    int rows, cols;
    float pixel_size_x;
    float pixel_size_y;
    
    // factory методы
    static ComplexField allocate(int rows, int cols);
    void free();
};