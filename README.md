**Задача**
Предсказать пол клиента по транзакциям

- Возможно ли предсказать пол клиента, используя сведения о поступлениях и тратах
- Какая точность такого предсказаниия

Необходимо предсказать вероятность пола 1 для каждого client_id, который присутствует в файле train.csv

arXiv:1911.02496

НАДО ПОДУМАТЬ

**Метрика - ROC-AUC**

**SberCloud**

DS Works

В zip архиве следующе файлы
- inference.py - результатом будет являться csv файл с предиктами  

- requirements.txt

- all additional files 

15 минут cpu


Best Parameters for CatBoostClassifier: {'classifier__depth': 6, 'classifier__iterations': 250, 'classifier__l2_leaf_reg': 4, 'classifier__learning_rate': 0.05}
Best ROC AUC Score for CatBoostClassifier: 0.8729944770233715
ROC AUC on Test Set for CatBoostClassifier: 0.7852669459662691

param_grid_cbc = {
    'classifier__iterations': [150, 200, 250],
    'classifier__learning_rate': [0.03, 0.05, 0.07],
    'classifier__depth': [6, 7, 9],
    'classifier__l2_leaf_reg': [2, 3, 4]
}


Взять все mcc и через них сгенерировать фиксированное количество фичей 