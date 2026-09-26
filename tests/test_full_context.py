import copy
import json
import pytest

from twinlab.benchmark_data import evidence_items, messages_for, tokenized_example
from twinlab.data import ROOT
from twinlab.download import MODEL_CONTEXT_TOKENS


class Tokenizer:
    eos_token = '<END>'
    def encode(self, text, **kwargs):
        return list(text.encode())
    def apply_chat_template(self, messages, **kwargs):
        return self.encode(json.dumps(messages))


def example():
    return {'pid':'p', 'id':'Target', 'group':'Ratio bias', 'context':[],
            'question':{'QuestionID':'QID196','QuestionType':'MC','QuestionText':'Choose','Options':['A','B']},
            'atoms':[{'levels':2,'value':1}]}


def test_complete_history_and_answer_masking_without_truncation():
    history=[{'BlockName':'History','Questions':[
        {'QuestionID':f'QID{i}','QuestionType':'TE','QuestionText':f'Question {i}',
         'Answers':{'Text':'past experience '*150+f'END_{i}'}} for i in range(12)]}]
    items=evidence_items(history)
    e=example();tok=Tokenizer()
    messages=messages_for(e,items)
    assert all(f'END_{i}' in messages[1]['content'] for i in range(12))
    result=tokenized_example(tok,e,items,MODEL_CONTEXT_TOKENS)
    assert result['prefix_length']>2048
    assert result['spans'][0][0]==result['prefix_length']
    changed=copy.deepcopy(e);changed['atoms'][0]['value']=2
    other=tokenized_example(tok,changed,items,MODEL_CONTEXT_TOKENS)
    assert result['ids'][:result['prefix_length']]==other['ids'][:other['prefix_length']]
    with pytest.raises(ValueError,match='no truncation'):
        tokenized_example(tok,e,items,128)


def test_free_text_multiselect_descriptions_and_zero_are_preserved():
    qs=[{'QuestionID':'Q1','QuestionType':'TE','QuestionText':'Explain','Answers':{'Text':'FREE_TEXT'}},
        {'QuestionID':'Q2','QuestionType':'MC','QuestionText':'Select','Options':['Red','Blue'],
         'Answers':{'SelectedText':['Red','Blue']}},
        {'QuestionID':'Q3','QuestionType':'DB','QuestionText':'DESCRIPTION'},
        {'QuestionID':'Q4','QuestionType':'Slider','QuestionText':'Quantity','Range':{'Min':0,'Max':100},
         'Answers':{'Values':[0]}}]
    items=evidence_items([{'BlockName':'History','Questions':qs}])
    content=messages_for(example(),items)[1]['content']
    assert 'FREE_TEXT' in content and 'DESCRIPTION' in content
    assert '"SelectedText":["Red","Blue"]' in content and '"Values":[0]' in content


def test_input_rejects_target_history_and_native_limit_is_unchanged():
    e=example()
    with pytest.raises(ValueError,match='Target question'):
        messages_for(e,[{'question':e['question']}])
    plan=json.loads((ROOT/'configs/model_plan.json').read_text())
    assert plan['max_sequence_tokens']==MODEL_CONTEXT_TOKENS==262144
