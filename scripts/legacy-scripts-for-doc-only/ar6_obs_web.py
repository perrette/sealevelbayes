import json
import sys
sys.path.append('notebooks')
from ar6.tables import ar6_table_9_8_medium_confidence, ar6_table_9_5

data = {
    'table_9_8': ar6_table_9_8_medium_confidence,
    'table_9_5': ar6_table_9_5,
}
json.dump(data, open('web/obs/ar6tables.json', 'w'))
