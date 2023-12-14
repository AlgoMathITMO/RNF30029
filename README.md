# РНФ 30029, 2023 г.

Описание модулей:
1. [Многомерное феноменологическое моделирование](#модуль-многомерное-феноменологическое-моделирование) 
2. [Многомасштабное моделирование](#модуль-многомасштабное-моделирование) (код в модуле 5)
3. [Инкрементальное обучение](#модуль-инкрементальное-обучение)
4. [Федеративное обучение](#модуль-федеративное-обучение)
5. [Муравьиная ферма](#модуль-муравьиная-ферма)
6. [Адаптированный генеративный дизайн](#модуль-aдаптированного-генеративного-дизайна)

Результаты работ в рамках РНФ 30029 2023 года послужили материалом для реализации агентной модели популяции потребителей с задаваемыми внешними условиями.
Все модули реализованы на языке Python и представлены в виде импортируемых библиотек или демонстрационных «блокнотов» Jupyter-Notebook. 
Могут использоваться отдельные процедуры из модулей, модули автономно, модули в комплексе, как, например, в модуле “Муравьиная ферма”. 
Взаимосвязь модулей соответсвует схеме связи результатов работы по проекту.

<p align="center" width="100%">
 <img src="https://github.com/AlgoMathITMO/RNF30029/blob/main/imgs/connection.png" width="80%" > 
<p align="center"><i>Связь результатов работы по проекту</i></p>
</p>  

### **Модуль “Многомерное феноменологическое моделирование”**
   
   Реализует идентификацию феноменологический модели потребительского поведения популяции и является дальнейшим развитием макромасштабной модели смены глобальных устойчивых состояний,
   обусловленной системным изменением экономики в силу наступления кризисных ситуаций, разработанной в 2022 году.
   Существенно переработан реализованный в модуле алгоритм анализа контекстной информации,
   в котором применены алгоритмы многоэтапного сентиментного анализа для оценки влияния содержания новостных сообщений на потребительское поведение.  
### **Модуль “Многомасштабное моделирование”**
  
   Включает разработанные ранее алгоритмы группировки потребителей по предсказуемости поведения (2021) и по времени реакции на критические события (2022);
   дополнен реализацией метода распознавания поведенческих стратегий в критические периоды и программной моделью,
   реализующая платёжное событие на микроуровне как обратную связь на изменения среды при изменении потребностей агента. 
### **Модуль “Инкрементальное обучение”**

   Включает процедуры инкрементального измерения предсказуемости поведения отдельных потребителей,
   программную реализацию алгоритма разделения популяции по предсказуемости и программную модель для прогнозирования суммарных трат в группах клиентов с различной предсказуемостью.
   Модуль представляет собой усовершенствование разработанного в 2021 году модуля динамической классификации предсказуемости поведения потребителей,
   в отличие от которого позволяет улучшать прогнозы поведения для всей популяции за счет учета изменения предсказуемости отдельных потребителей в критических условиях. 
### **Модуль “Федеративное обучение”**

   Осуществляет методически организованное федеративногое обучение для прогнозирования переходных процессов и включает модели, обучаемые на стороне источника данных и блок,
   реализующий прогностическое моделирование целевого временного ряда с учётом полученных параметров моделей, обученных на сторонних данных. 
### **Модуль “Муравьиная ферма”**

   Представляет собой реализацию агентной модели потребительского поведения популяции в периоды кризисов.
   Содержит процедуры, выполняющие агрегацию всех прежде разработанных модулей, позволяющая осуществлять воспроизведение процессов в микро- и макромасштабах,
   как при реализации заданных сценариев, так и при применении реальных данных. В модуле использованы процедуры из модулей
   “Многомерное феноменологическое моделирование”, “Многомасштабное моделирование”, “Инкрементальное обучение”. 
   
### **Модуль aдаптированного генеративного дизайна**

   Включает программную реализацию метода TRGAN для синтеза данных дискретных финансовых процессов на основе реальных транзакционных дата-сетов, 
   математической модели и результатов предсказательного моделирования. В этот же модуль входят процедуры, реализующие метод моделирования городских кварталов.  
