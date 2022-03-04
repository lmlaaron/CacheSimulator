import sys
import pandas as pd
from cache_guessing_game_env_impl import CacheGuessingGameEnv
from pandas.core.arrays import numeric

class CategorizationParser:
  def __init__(self,filename):
    self.filename = filename
    self.number_of_set = 2

  def readfile(self):# python categorization_parser.py temp.txt
    patterns=[]
    f = open(self.filename, mode='r', encoding='UTF-8')
    lines = f.readlines()
    for d in lines:
      d = d.split()
      d = [int(i) for i in d]
      patterns.append(d)
    return patterns

  def parse_action(self, action): 
    gameenv = CacheGuessingGameEnv() 
    action = gameenv.parse_action(action)
    return action

  def convert_dataframe(self, input): # split into [(attacker's)addr, is_guess, is_victim, is_flush, victim_addr]
    df = pd.DataFrame(input)  
    df = df.astype('int')
    df.columns =['addr', 'is_guess', 'is_victim', 'is_flush', 'victim_addr']
    return df

  def get_set(self, row): #return addr%self.number_of_set
    
    #df.addr = df.addr%self.number_of_set
    return row['addr']%self.number_of_set

  def add_setcolumn(self, df):
    #df['set'] = df.apply(self, get_set(df.addr)) 
    df['set'] = df.apply (lambda row: self.get_set(row), axis=1)
    return df
    

  def get_order(self, array): #return order_array
    pass 

  def rename_column(self,df):
    #set0 = df[df.set==0]
    #set0['order'] = set0['addr'].rank(method='dense',ascending=False).astype(int)
    #set1 = df[df.set==1]
    #set1['order'] = set1['addr'].rank(method='dense',ascending=False).astype(int)
    df['order'] = df['addr'].rank(method='dense',ascending=False).astype(int)
    #frames = [set0, set1]
    #result = pd.concat(frames)
    #output = pd.DataFrame(result)
    #output =df.sort_index(axis=0, ascending=True)
    return df

  def remove_rep(self, pattern): # return pattern.drop_duplicates()
    pass

  def main_parser(self, pattern):

    for action in pattern :
      action_parsed = categorization_parser.parse_action(action)
      pattern_parsed.append(action_parsed)

    df = categorization_parser.convert_dataframe(pattern_parsed)
    df = categorization_parser.add_setcolumn(df)
    df = categorization_parser.rename_column(df)
    output = df.values.tolist()
    return output

def main(argv): # Defining main function
  filename = argv[1]
  print(filename)

  categorization_parser = CategorizationParser(filename)
  patterns = categorization_parser.readfile()
  print(patterns)
  
  for pattern in patterns :
    pattern_parsed = []
    for action in pattern :
      action_parsed = categorization_parser.parse_action(action)
      pattern_parsed.append(action_parsed)
    print(pattern_parsed)
  
  #pattern_parsed = categorization_parser.parse_action(action)

  df = categorization_parser.convert_dataframe(pattern_parsed)
  df = categorization_parser.add_setcolumn(df)
  print(df)
  df = categorization_parser.rename_column(df)
  
  print(df)
  output = categorization_parser.main_parser(df)
  print(output)

if __name__=="__main__": # Using the special variable
    main(sys.argv)