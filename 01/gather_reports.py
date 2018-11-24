import os
import sys
from shutil import copy2
from shutil import copy2

# copy2('test/test.py', 'test1/test1.py')

test = [x for x in os.walk('NN_summary')]

# for nn in test[0][1]:
#     print(nn)

for network in test[0][1]:
    copy2("NN_summary/"+network+'/report_template.pdf', "reports/"+network+'.pdf')

