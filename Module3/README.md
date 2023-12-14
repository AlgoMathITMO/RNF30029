# Dyn_Eval_of_Pred
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Mpkosh/Dyn_Eval_of_Pred/blob/main/Method.ipynb)

## Метод динамической оценки предсказуемости 
Для определения моделей переходных процессов был создан модуль инкрементального обучения для идентификации моделей переходных финансовых процессов, позволяющий динамически оценивать предсказуемость последовательности транзакций клиентов банка с целью преодоления снижения оценки прогнозирования на временном промежутке. Ключевыми особенностями применённого метода являются:
* динамическое измерение предсказуемости поведения клиентов на микроуровне для учета переходных процессов,
* раделение клиентов по классам предсказуемости с учетом внутренней или реализованной предсказуемостей,
* прогнозирование всей популяции на основе заданного класса клиентов с целью улучшения качества прогноза. 

<p align="center" width="100%">
 <img src="https://github.com/Mpkosh/Dyn_Eval_of_Pred/blob/main/imgs/Алгоритм.png" width="70%" > 
<p align="center"><i>Схема работы алгоритма</i></p>
</p>  
 
## Описание файлов
* preprocessing.py - функции для предобработки данных,
* micro_help_functions.py - вспомогательные функции для работы с методом на микроуровне,
* micro_model_functions.py - функции для работы с моделью на микроуровне,
* macro_help_functions.py - вспомогательные функции для работы с методом на мезо- и макроуровне,
* macro_model_functions.py - функции для работы с моделью на мезо- и макроуровне.

## Результаты
Качество прогноза всех клиентов было улучшено:
* у базовых моделей на **5.88 процентных пункта** в случае медианы MAPE (от 12.84% к 6.96%) и **0.19 пункта** в случае доли попадания (от 0.19 к 0.38); 
* у инкрементальных моделей на **0.57 процентных пункта** в случае медианы MAPE (от 5% к 4.43%) и **0.06 пункта** в случае доли попадания (от 0.5 к 0.56).