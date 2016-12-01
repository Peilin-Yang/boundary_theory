query = [ 
  {
    "collection": "disk12_nostopwords",
    "collection_formal_name": "disk12",
    "cnt": 150,
    "qf_parts": ['title'],
    "qrel_program": 'trec_eval -m all_trec -q',
    "main_metric": "MAP"
  },
  { 
    "collection": "disk45_nostopwords", 
    "collection_formal_name": "disk45",
    "cnt": 250,
    "qf_parts": ['title'],
    "qrel_program": 'trec_eval -m all_trec -q',
    "main_metric": "MAP"
  },  
  {
    "collection": "wt2g_nostopwords",
    "collection_formal_name": "WT2G",
    "cnt": 50,
    "qf_parts": ['title'],
    "qrel_program": 'trec_eval -m all_trec -q',
    "main_metric": "MAP"
  },
  {
    "collection": "wt10g_nostopwords",
    "collection_formal_name": "WT10G",
    "cnt": 100,
    "qf_parts": ['title'],
    "qrel_program": 'trec_eval -m all_trec -q',
    "main_metric": "MAP"
  },
  {
    "collection": "gov2_nostopwords",
    "collection_formal_name": "GOV2",
    "cnt": 150,
    "qf_parts": ['title'],
    "qrel_program": 'trec_eval -m all_trec -q',
    "main_metric": "MAP"
  },
  {
    "collection": "aquaint_nostopwords",
    "collection_formal_name": "AQUAINT",
    "cnt": 50,
    "qf_parts": ['title'],
    "qrel_program": 'trec_eval -m all_trec -q',
    "main_metric": "MAP"
  }
]
