import os.path
import sys

import pandas as pd
import numpy as np

from psycopg2 import connect, sql
from sklearn.model_selection import train_test_split, GridSearchCV
from catboost import CatBoostRegressor

from sdv.metadata import SingleTableMetadata
from sdv.single_table import CTGANSynthesizer
from sdv.sampling import Condition

def train_regressor(kohorte_train_features, kohorte_train_label, model_name):
	categorical_features = np.where(kohorte_train_features.dtypes != float)[0]

	grid = {
		'learning_rate': [0.03, 0.1, 0.5, 0.8, 1],
		'depth': [4, 6, 8],
		'l2_leaf_reg': [10, 30, 50, 80, 100],
		'has_time': [True, False],
	}

	model = CatBoostRegressor(
		iterations = 1000,
		cat_features = categorical_features,
		task_type = 'GPU',
		verbose = 0,
		random_seed = 42,
		od_type = 'Iter',
		od_wait = 50
	)

	search = GridSearchCV(
		estimator = model,
		param_grid = grid,
		n_jobs = 1,
		refit = True,
		cv = 3,
		error_score = 'raise'
	)

	search.fit(X = kohorte_train_features, y = kohorte_train_label)
	search.best_estimator_.save_model('models/' + model_name)
	return search.best_estimator_

icd_code_groups = {
	'^R10': {
		'icd_code': ['R100', 'R1010', 'R1011', 'R1012', 'R1013', 'R102', 'R1030', 'R1031', 'R1032', 'R1033', 'R10811', 'R10813', 'R10816', 'R10817', 'R10819', 'R10821', 'R1084', 'R109']
	},
	'^R07': {
		'icd_code': ['R070', 'R071', 'R072', 'R0781', 'R0789', 'R079']
	},
	'^J18': {
		'icd_code': ['J180', 'J188', 'J189']
	},
	'^R06': {
		'icd_code': ['R0600', 'R0601', 'R0602', 'R0603', 'R0609', 'R061', 'R062', 'R066', 'R0681', 'R0682', 'R0689', 'R069']
	},
	'^R50': {
		'icd_code': ['R502', 'R5082', 'R5084', 'R509']
	},
	'^K92': {
		'icd_code': ['K920', 'K921', 'K922', 'K9289', 'K929']
	},
	'^R53': {
		'icd_code': ['R530', 'R531', 'R5381', 'R5383']
	},
	'^R41': {
		'icd_code': ['R410', 'R412', 'R413', 'R4181', 'R4182', 'R4183', 'R4189', 'R419']
	},
	'^R07.*[^89]$': {
		'icd_code': ['R070', 'R071', 'R072', 'R0781']
	},
	'^F32.*[^89]$': {
		'icd_code': ['F322', 'F323']
	},
	'^R50.*[^89]$': {
		'icd_code': ['R502', 'R5082', 'R5084']
	},
	'^R10.*[^89]$': {
		'icd_code': ['R100', 'R1010', 'R1011', 'R1012', 'R1013', 'R102', 'R1030', 'R1031', 'R1032', 'R1033', 'R10811', 'R10813', 'R10816', 'R10817', 'R10821', 'R1084']
	},
	'^R06.*[^89]$': {
		'icd_code': ['R0600', 'R0601', 'R0602', 'R0603', 'R061', 'R062', 'R066', 'R0681', 'R0682']
	},
	'^K92.*[^89]$': {
		'icd_code': ['K920', 'K921', 'K922']
	},
	'^R53.*[^89]$': {
		'icd_code': ['R530', 'R531', 'R5381', 'R5383']
	},
	'^R41.*[^89]$': {
		'icd_code': ['R410', 'R412', 'R413', 'R4181', 'R4182', 'R4183']
	}
}

if __name__ == '__main__':
	case = sys.argv[1]

	conn = connect(
		dbname = 'dbname',
		user = 'user',
		password = 'password'
	)
	cursor = conn.cursor()

	cursor.execute(
		sql.SQL(
			'SELECT los, admission_location, icd_code, anchor_age, insurance, ethnicity, gender, resprate, sbp, pain, diag_count, med_count, ed_los, prev_stay_avg_dur FROM alexander WHERE icd_code = {icd_code} ORDER BY subject_id, hadm_id'
		).format(
			icd_code = sql.Literal(case)
		)
	)
	kohorte = pd.DataFrame(cursor.fetchall(), columns = ['los', 'admission_location', 'icd_code', 'anchor_age', 'insurance', 'ethnicity', 'gender', 'resprate', 'sbp', 'pain', 'diag_count', 'med_count', 'ed_los', 'prev_stay_avg_dur'])
	for column in ['los', 'anchor_age', 'resprate', 'sbp', 'pain', 'diag_count', 'med_count', 'ed_los', 'prev_stay_avg_dur']:
		kohorte[column] = kohorte[column].astype(float)
	kohorte_features = kohorte.drop('los', axis = 1)
	kohorte_label = kohorte.los
	kohorte_train_features, kohorte_test_features, kohorte_train_label, kohorte_test_label = train_test_split(kohorte_features, kohorte_label, test_size = 0.3, random_state = 42)

	if os.path.exists('models/' + case + '_real'):
		model = CatBoostRegressor()
		model.load_model('models/' + case + '_real')
	else:
		model = train_regressor(kohorte_train_features, kohorte_train_label, case + '_real')

	errors = list()
	predictions = model.predict(kohorte_test_features)
	for index, actual_los in enumerate(kohorte_test_label):
		errors.append(abs(actual_los - predictions[index]))

	with open('evaluation.csv', 'a') as fw:
		fw.write(case + ',' + str(sum(errors) / len(errors)) + ',')

	cursor.execute(
		sql.SQL(
			'SELECT los, admission_location, icd_code, anchor_age, insurance, ethnicity, gender, resprate, sbp, pain, diag_count, med_count, ed_los, prev_stay_avg_dur FROM alexander WHERE icd_code != {icd_code} ORDER BY subject_id, hadm_id'
		).format(
			icd_code = sql.Literal(case)
		)
	)
	rest = pd.DataFrame(cursor.fetchall(), columns = ['los', 'admission_location', 'icd_code', 'anchor_age', 'insurance', 'ethnicity', 'gender', 'resprate', 'sbp', 'pain', 'diag_count', 'med_count', 'ed_los', 'prev_stay_avg_dur'])
	for column in ['los', 'anchor_age', 'resprate', 'sbp', 'pain', 'diag_count', 'med_count', 'ed_los', 'prev_stay_avg_dur']:
		rest[column] = rest[column].astype(float)

	kohorte_train = pd.merge(kohorte_train_features, kohorte_train_label, left_index = True, right_index = True)
	without_test = pd.concat([rest, kohorte_train])



	without_test_train_features = without_test.drop('los', axis = 1)
	without_test_train_label = without_test.los
	if os.path.exists('models/' + case + '_all'):
		model = CatBoostRegressor()
		model.load_model('models/' + case + '_all')
	else:
		model = train_regressor(without_test_train_features, without_test_train_label, case + '_all')

	errors = list()
	predictions = model.predict(kohorte_test_features)
	for index, actual_los in enumerate(kohorte_test_label):
		errors.append(abs(actual_los - predictions[index]))
	with open('evaluation.csv', 'a') as fw:
		fw.write(str(sum(errors) / len(errors)) + ',')



	metadata = SingleTableMetadata.load_from_json(filepath = 'metadata.json')
	metadata.validate()

	if os.path.exists('models/' + case + '.pkl'):
		synthesizer = CTGANSynthesizer.load(filepath = 'models/' + case + '.pkl')
	else:
		synthesizer = CTGANSynthesizer(
			metadata,
			enforce_rounding = False,
			epochs = 300,
			verbose = True
		)

		synthesizer.fit(data = without_test)
		synthesizer.save(filepath = 'models/' + case + '.pkl')

	if os.path.exists('samples/' + case + '.csv'):
		synthetic_data = pd.read_csv('samples/' + case + '.csv')
	else:
		Condition(
			num_rows = 4000,
			column_values = {'icd_code': case}
		)
		synthetic_data = synthesizer.sample_from_conditions(
			conditions = [condition],
			batch_size = 4000,
			max_tries_per_batch = 200000,
			output_file_path = 'samples/' + case + '.csv'
		)

	synthetic = pd.concat([kohorte_train, synthetic_data])

	synthetic_features = synthetic.drop('los', axis = 1)
	synthetic_label = synthetic.los
	synthetic_train_features, synthetic_test_features, synthetic_train_label, synthetic_test_label = train_test_split(synthetic_features, synthetic_label, test_size = 0.3, random_state = 42)

	if os.path.exists('models/' + case + '_synthetic'):
		model = CatBoostRegressor()
		model.load_model('models/' + case + '_synthetic')
	else:
		model = train_regressor(synthetic_train_features, synthetic_train_label, case + '_synthetic')

	errors = list()
	predictions = model.predict(kohorte_test_features)
	for index, actual_los in enumerate(kohorte_test_label):
		errors.append(abs(actual_los - predictions[index]))
	with open('evaluation.csv', 'a') as fw:
		fw.write(str(sum(errors) / len(errors)) + '\n')

	conn.close()
