import glob
import shutil
import os

test_cases = glob.glob('./output/*')

cases_to_del = ['beam', 'aero', 'beam_modal_analysis', 'WriteVariablesTime', 'stability', 'frequencyresponse']

for case_dir in test_cases:
	for postproc in cases_to_del:
		target = case_dir + '/' + postproc
		if os.path.isdir(target):
			shutil.rmtree(target)
