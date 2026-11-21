import pandas as pd


path = '/data3/yxfeng/datasets/OpenR1-Math-220k/extended/train-00000-of-00010.parquet'

df = pd.read_parquet(path)

'''

Index(['problem', 'solution', 'answer', 'problem_type', 'question_type',
       'source', 'uuid', 'is_reasoning_complete', 'generations',
       'correctness_math_verify', 'correctness_llama', 'finish_reasons',
       'correctness_count', 'messages'],
      dtype='str')

      
question_ls = df['problem'].tolist()
answer_ls = df['answer'].tolist()

'''

breakpoint()


