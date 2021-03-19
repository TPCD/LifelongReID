from __future__ import print_function, absolute_import

from .test_continual_operation_neck import plot_prerecall_curve, test_continual_neck, fast_test_continual_neck, output_featuremaps_from_fixed
from .test_incremental_operation_metagraph_graphfd import fast_test_incremental_metagraph_graphfd, save_and_fast_test_incremental_metagraph_graphfd
from .train_incremental_operation_metagraph_fd import train_incremental_metagraph_graphfd_an_epoch, train_incremental_metagraph_graphfd_no_detach_an_epoch