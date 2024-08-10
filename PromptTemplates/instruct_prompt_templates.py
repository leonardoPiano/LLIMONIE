NER='''\
#NER Given a sentence find all the named entities inside it. For each mention assign two types one specific and another more generic (hyperonym).
Your answer is a list of entities formatted as:
Entities: [mention[type1|type2],..]
'''
RE="""\
#RE Given a text and a list of named entities extracted from it, find the relationships between those entities.
For each relationship you find construct a triple formed as Subject[type1|type2];predicate;Object[type1|type2].
Structure the output as a numbered list, placing one triple per line.
"""
JOINT="""\
#JOINT Given a text, find all the semantic triplet inside it.
Analyze the text and identify the entities first, for each entity you have to assign two categories, one specific and another more generic (hyperonym).
Each entity will then have the form mention[type1|type2].
Then find the relationship that exist between the identified entities.
Each triple will have only one predicate. If the same entity, has multiple relations, generate a separate triple.
The triple will have this form Subject[type1|type2];predicate;Object[type1|type2]. 
Structure the output as a numbered list, placing one triple per line. 
"""
