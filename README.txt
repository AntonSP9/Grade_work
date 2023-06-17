Описание задачи:

Даны данные ("ga_hits-002.csv" и "ga_sessions.csv") по посещению пользователями сайта.

Целевое действие — события типа «Оставить заявку» и «Заказать звонок» 
(ga_hits.event_action in ['sub_car_claim_click', 'sub_car_claim_submit_click', 
'sub_open_dialog_click', 'sub_custom_question_submit_click', 
'sub_call_number_click', 'sub_callback_submit_click', 'sub_submit_success', 
'sub_car_request_submit_click']).

Задача: 

- Построить модель, предсказывающую совершение целевого действия 
(ориентировочное значение ROC-AUC ~ 0.65) — факт совершения 
пользователем целевого действия;

- Упаковать получившуюся модель в сервис, который будет брать на 
вход все атрибуты, типа utm_*, device_*, geo_*, и отдавать на выход 
0/1 (1 — если пользователь совершит любое целевое действие).

Работа с данными и обучение модели - файл "Final_work.ipynb";
API - "main.py" в папке "Final_work_API"
