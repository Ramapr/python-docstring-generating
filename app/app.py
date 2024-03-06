from transformers import AutoTokenizer, AutoModelForCausalLM
import sys
import streamlit as st
from streamlit.web import cli as stcli
from streamlit import runtime
from parse import get_first_params

AVAILABLE_ROWS = ["Parameters", "Notes", "Raises", "Example"]

template = '''<|endoftext|>
%s

# docstring
"""'''

add_row = "\n\n    %s\n    ----------"

wpath = "./model"


@st.cache_resource  # 👈 Add the caching decorator
def load_mdl_tok(pth):
    tokenizer = AutoTokenizer.from_pretrained(wpath)  
    model = AutoModelForCausalLM.from_pretrained(wpath)
    return model, tokenizer


def process(inputs, model, tokenizer, doc_max_length = 128):
    generated_ids = model.generate(
        **inputs,
        max_length=inputs.input_ids.shape[1] + doc_max_length,
        do_sample=False,
        return_dict_in_generate=True,
        num_return_sequences=1,
        output_scores=True,
        pad_token_id=50256,
        eos_token_id=50256
    )
    return tokenizer.decode(generated_ids.sequences[0], skip_special_tokens=False)


def main():
    model, tokenizer = load_mdl_tok(wpath)

    st.header("DOC-string generator")
    text = st.text_area(label="input text below", 
                        height=350, 
                        value="example string")
  
    selected_opt = st.multiselect("choose variables:", AVAILABLE_ROWS, default="Parameters")
    but = st.button('generate')
    if but:
        if len(text) < 10:
            st.error('https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcS21LyBLTIs3fStzfUhWxCj38qA2rvMWjegXC_gI00NcQfoIu7rZSaQFL-KC1mvWVA8F4o&usqp=CAU', icon="🚨")
          
        req = template % text
        with st.spinner('Wait for it...'):
            msg = process(tokenizer(req, return_tensors='pt'), model, tokenizer)
            my_bar = st.progress(0, text="Generate for selected variables. Please wait...")
            for pi, line in enumerate(selected_opt):
                msg = process(tokenizer(msg[: -25] + add_row % line,  return_tensors='pt'), model, tokenizer)
                my_bar.progress((pi + 1) / len(selected_opt), text="Operation in progress. Please wait.")
            st.code(msg[13:-13], language="python")


if __name__ == '__main__':
    if runtime.exists():
        main()
    else:
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())

