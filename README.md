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



Best Parameters for CatBoostClassifier: {'classifier__max_depth': 6, 'classifier__n_estimators': 1000}
Best ROC AUC Score for CatBoostClassifier: 0.8739093308426327
ROC AUC on Test Set for CatBoostClassifier: 0.7869260288474262

Best Parameters for RandomForestClassifier: {'classifier__max_depth': 30, 'classifier__min_samples_leaf': 2, 'classifier__min_samples_split': 5, 'classifier__n_estimators': 500, 'classifier__verbose': 2}
Best ROC AUC Score for RandomForestClassifier: 0.8486493093932985
ROC AUC on Test Set for RandomForestClassifier: 0.7576948524546778

Best Parameters for SVClassifier: {'classifier__C': 0.1, 'classifier__gamma': 'scale', 'classifier__kernel': 'linear', 'classifier__verbose': True}
Best ROC AUC Score for SVClassifier: 0.8302128318322157
ROC AUC on Test Set for SVClassifier: 0.7415561730845573


Voting 0.8675169791010107

model_rf = RandomForestClassifier(max_depth=30, min_samples_leaf=2, min_samples_split=5, n_estimators=500)
model_cb = CatBoostClassifier(depth=6, iterations=1500, l2_leaf_reg=4, learning_rate=0.05)
model_svc = SVC(probability=True, C=0.1, gamma= 'scale', kernel= 'linear')