import xml.etree.ElementTree as ET
import re
import pandas as pd
from collections import Counter
import os

directory = 'Scientist_Testdata/XML-Projects'
mix_packs_path = 'Scientist_Testdata/Mix-Packs/'

# function to find the frequency of elements of a list
def most_used(my_list):
    ct = Counter(my_list)
    max_value = max(ct.values())
    return sorted(key for key, value in ct.items() if value == max_value)

# global variables for displaying global statistics of the project
global_dict = {}
parsed_xmls_list = []
used_mix_pack = []
bpm_list = []

for filename in os.listdir(directory):
    if filename.endswith(".xml"):
        xml_path = os.path.join(directory, filename)

        tree = ET.parse(xml_path)

        # local variables for storing local statistics of the project
        my_track_dict = {}
        loop_counter = 0
        number_of_parts_counter = 0
        loop_composition_and_status = []
        used_mix_pack_loops = []

        root = tree.getroot()
        my_track_dict['bpm'] = int(root.attrib['bpm'])

        for child in root:
            if(child.tag != 'parts'):
                continue
            for child in child:
                if (child.tag != 'part'):
                    continue
                number_of_parts_counter = number_of_parts_counter + 1
                for child in child:
                    if (child.tag != 'channels'):
                        continue
                    for child in child:
                        if (child.tag != 'channel'):
                            continue
                        if(child.attrib['loop'] != '' and child.attrib['is_active'] != 'false' and re.match('(mmj://styles/id/)(ST_Electro_Lite|ST_Electro_Pop|ST_House_Lite|ST_Loudly_Exclusive)(/.+)', child.attrib['loop'])):
                            loop_counter = loop_counter + 1
                            loop_composition = re.sub('mmj://styles/id/', '', child.attrib['loop'])
                            try:
                                used_loop_path = mix_packs_path + loop_composition + '.csv'
                                desc_csv = pd.read_csv(used_loop_path)
                            except:
                                pass
                            try:
                                used_loop_path = mix_packs_path + loop_composition + ' 1' + '.csv'
                                desc_csv = pd.read_csv(used_loop_path)
                            except:
                                pass
                            try:
                                used_loop_path = mix_packs_path + loop_composition + '  1' + '.csv'
                                desc_csv = pd.read_csv(used_loop_path)
                            except:
                                pass

                            new_label = desc_csv[['Label', 'd1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7', 'd8']].values[0]
                            new_label_str = ''
                            for i in range(len(new_label)):
                                new_label_str = new_label_str + ' ' + str(new_label[i])

                            loop_composition_and_status.append((new_label_str[1: -1], child.attrib['is_active']))
                            loop_composition_path = loop_composition.split('/')
                            mix_pack_loop_used = loop_composition_path[2]
                            mix_pack_used = loop_composition_path[0]

                            used_mix_pack_loops.append(mix_pack_loop_used)
                            used_mix_pack.append(mix_pack_used)

        my_track_dict['number_of_parts'] = int(number_of_parts_counter)
        my_track_dict['number_of_loops_in_parts'] = int(loop_counter)
        my_track_dict['most_used_loops'] = most_used(used_mix_pack_loops)
        my_track_dict['features'] = loop_composition_and_status

        parsed_xmls_list.append(my_track_dict)

global_dict['most_used_mix_pack'] = most_used(used_mix_pack)
global_dict['most_informative_part_indices'] = ('loop', 'bpm')
global_dict['parsed_xmls'] = parsed_xmls_list

# the global dict contains a consolidated list of all parsed xmls and global statistics, and with the xmls as dicttionaries with local statistics
# keys are 'most_used_mix_pack', 'most_informative_part_indices', 'parsed_xmls' for global dict 'global_dict'
# keys are 'number_of_parts', 'number_of_loops_in_parts' and 'most_used_loops' for local dict of each parsed xml

import numpy as np

xml_list = global_dict['parsed_xmls']

features = np.empty((len(xml_list), 3))
for i in range(len(xml_list)):
    features[i][0] = xml_list[i]['bpm']
    features[i][1] = xml_list[i]['number_of_parts']
    features[i][2] = xml_list[i]['number_of_loops_in_parts']

np.save('my_feature_array.npy', features)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

feature_array = np.load('my_feature_array.npy')

scaler = MinMaxScaler()
scaled_feature_array = scaler.fit_transform(feature_array)

kmeans_estimator = KMeans(n_clusters=3)
y_kmeans_estimator = kmeans_estimator.fit_predict(scaled_feature_array)

cosine_sim = cosine_similarity(scaled_feature_array)

f, (ax1, ax2) = plt.subplots(1, 2)
ax1.scatter(scaled_feature_array[:, 0], scaled_feature_array[:, 1], c=y_kmeans_estimator, cmap='rainbow')
ax1.set_title('K-Means Clustering')
ax2.imshow(cosine_sim)
ax2.set_title('Cosine Similarity')
plt.show()