from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


def run_query(query, records, global_state=None):
  if not global_state:
    global_state = query.initial_global_state()
  initial_params = query.derive_initial_sample_params(global_state)
  params = query.derive_sample_params(global_state)
  sample_state = query.initial_sample_state(params=initial_params,
                                            template=next(iter(records)))
  for record in records:
    sample_state = query.accumulate_record(params, sample_state, record)
  return query.get_noised_result(sample_state, global_state)
