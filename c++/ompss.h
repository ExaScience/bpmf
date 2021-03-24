

void sample_task(
    long iter,
    long idx,
    const HyperParams *hp_ptr, 
    const double *other_ptr,
    const double *ratings_ptr,
    const int *inner_ptr,
    const int *outer_ptr,
    double *items_ptr);

void sample_task_scheduler(
    int from,
    int to,
    int num_ratings,
    int outer_size_plus_one,
    int num_items,
    int other_num_items,
    const double *other_ptr,

    const int this_iter,
    const HyperParams *this_hp_ptr,
    const int *this_inner_ptr,
    const int *this_outer_ptr,
    const double *this_ratings_ptr,
    double *this_items_ptr
);