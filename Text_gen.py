import streamlit as st
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config
import textwrap

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def main():
    with st.sidebar:
        add_radio = st.radio("Choose a model!",("Project information","Text Generate"))
    
    if add_radio == 'Project information':
        st.header("Генерация текста с помощью нейросети")
        st.write('Обработка естественного языка. Скажу честно — это просто не моя тема,\
            но тут не моё мнение. Тут есть и твоё личное мнение. Я буду его озвучивать и даже \
                если мне оно скажут — «да нет, так нельзя» или «ты сам себе хозяин», пусть, я не \
                    против.  Я против того, кто не хочет учиться. Обработка естественного языка — \
                        это я. И если у кого то не хватает сил и не хватает денег и здоровья для \
                            того чтобы увидеть мир изнутри и остаться внутри и понять меня, что \
                                ты внутри меня. Я не боюсь говорить подобным. Ты! Я! В тебе! Не в себе! \
                                    Я чувствую себя чужаком..... А кто же я такой для них????????')

    
    elif add_radio =='Text Generate':
        st.header("Генерация текста с помощью нейросети")
        st.write('Это не самый простой способ сделать свой блог уникальным. В этой статье я\
            буду говорить только о том, какие инструменты есть в интернете для\
                обработки естественного языка. Если вам не нравятся ваши посты вы можете\
                использовать мой блог. И так, чтобы ваши статьи не выглядели скучными в \
                    глазах моих знакомых или ваших читателей, я покажу, какие способы \
                        обработки использовал я, как и где вы можете найти интересные статьи, поможет мой GitHub.')
        title = st.text_input(label = 'Write your text', value =' ')
        length = st.slider('max_length', min_value=50, max_value=150, step=10,value =100, help='Рекомендуем устанавливать значение 100. Длина предложения.')
        beam = st.slider('Num_beam?', min_value=1, max_value=7, step=1,value=4, help='Рекомендуем устанавливать значение 4. Поиск луча — это алгоритм, используемый во многих моделях НЛП \
            и распознавания речи в качестве окончательного уровня принятия решений для выбора наилучшего вывода с заданными целевыми переменными, \
            такими как максимальная вероятность или следующий выходной символ.')
        temp = st.slider('temperature', min_value=10., max_value=20., step=1.,value=18.,help='Рекомендуем устанавливать значение 18. Температура повышает шансы появления наиболее вероятных токенов, \
            снижая при этом шансы появления неподходящих. ')
        k = st.slider('top_k', min_value=10, max_value=100, step=10,value=50, help = 'Эксперементальный параметр. Изменение параметра top-k устанавливает размер короткого списка, \
            из которого модель производит выборку при выводе каждого токена.')
        p = st.slider('top_p', min_value=0.1, max_value=1.0, step=0.1,value=0.6,help = 'Эксперементальный параметр. top-k открывает двери для популярной стратегии декодирования, которая динамически устанавливает размер короткого списка токенов. \
            Этот метод, называемый Nucleus Sampling , составляет короткий список лучших токенов, сумма вероятностей которых не превышает определенного значения')
        sequences = st.slider('num_return_sequences', min_value=1, max_value=5, step=1, help='Количество абзацев')
        no_repeat = st.slider('no_repeat_ngram_size', min_value=1, max_value=5, step=1,value=2, help='Рекомендуем устанавливать значение 2. Штраф за n-граммы гарантирует, \
            что ни одна n-грамма не появится дважды')
        char_in_str = st.slider('char_in_str', min_value=10, max_value=100, step=10,value=30, help='Рекомендуем устанавливать значение 30. Количество символов в предложении.')
        if title == ' ':
            st.write(' ')
        else:
            model = GPT2LMHeadModel(GPT2Config(n_positions=2048))
            model.transformer.wte = torch.nn.Embedding(50264, 768)
            model.lm_head = torch.nn.Linear(768, 50264, bias=False)
            model.load_state_dict(torch.load('models/sebrModel.pt', map_location=torch.device('cpu')))
            tokenizer = GPT2Tokenizer.from_pretrained('sberbank-ai/rugpt3small_based_on_gpt2')
            title = tokenizer.encode(title, return_tensors='pt')#.to('device')
            out = model.generate(
                input_ids=title,
                max_length=length,
                num_beams=beam,
                do_sample=True,
                temperature=temp,
                top_k=k,
                top_p=p,
                no_repeat_ngram_size=no_repeat,
                num_return_sequences=sequences,
                ).numpy()#.cpu()
            for out_ in out:
                st.write(textwrap.fill(tokenizer.decode(out_), char_in_str), end='\n------------------\n')

if __name__ == '__main__':
    main()