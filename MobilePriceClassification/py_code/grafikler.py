import pandas as pd
import plotly as px

data = pd.read_csv('train.csv')
data.head()

import plotly
import plotly.express as px

#############   Bluetooth ###############################
fig = px.pie(data['blue'].value_counts().reset_index(), values = 'blue',
             names = ['No', 'Yes'])
fig.update_traces(textposition = 'inside', 
                  textinfo = 'percent + label', 
                  hole = 0.6, 
                  marker = dict(colors = ['#C2A7B5','#DE1A82'],
                                line = dict(color = 'white', width = 3)))

fig.update_layout(annotations = [dict(text = 'Bluetooth', 
                                      x = 0.5, y = 0.5,
                                      font_size = 24, showarrow = False, 
                                      font_family = 'Verdana',
                                      font_color = 'black')],
                  showlegend = False)
                  
fig.show()

#############   ÇİFT SİM ###############################
fig = px.pie(data['dual_sim'].value_counts().reset_index(), values = 'dual_sim',
             names = ['No', 'Yes'])
fig.update_traces(textposition = 'inside', 
                  textinfo = 'percent + label', 
                  hole = 0.6, 
                  marker = dict(colors = ['#C2A7B5','#DE1A82'],
                                line = dict(color = 'white', width = 3)))

fig.update_layout(annotations = [dict(text = 'Çift Sim', 
                                      x = 0.5, y = 0.5,
                                      font_size = 24, showarrow = False, 
                                      font_family = 'Verdana',
                                      font_color = 'black')],
                  showlegend = False)
                  
fig.show()

#############   4G ###############################
fig = px.pie(data['four_g'].value_counts().reset_index(), values = 'four_g',
             names = ['No', 'Yes'])
fig.update_traces(textposition = 'inside', 
                  textinfo = 'percent + label', 
                  hole = 0.6, 
                  marker = dict(colors = ['#C2A7B5','#DE1A82'],
                                line = dict(color = 'white', width = 3)))

fig.update_layout(annotations = [dict(text = '4G', 
                                      x = 0.5, y = 0.5,
                                      font_size = 24, showarrow = False, 
                                      font_family = 'Verdana',
                                      font_color = 'black')],
                  showlegend = False)
                  
fig.show()

#############   3G ###############################
fig = px.pie(data['three_g'].value_counts().reset_index(), values = 'three_g',
             names = ['No', 'Yes'])
fig.update_traces(textposition = 'inside', 
                  textinfo = 'percent + label', 
                  hole = 0.6, 
                  marker = dict(colors = ['#C2A7B5','#DE1A82'],
                                line = dict(color = 'white', width = 3)))

fig.update_layout(annotations = [dict(text = '3G', 
                                      x = 0.5, y = 0.5,
                                      font_size = 24, showarrow = False, 
                                      font_family = 'Verdana',
                                      font_color = 'black')],
                  showlegend = False)
                  
fig.show()


#############   DOKUNMATİK EKRAN ###############################
fig = px.pie(data['touch_screen'].value_counts().reset_index(), values = 'touch_screen',
             names = ['No', 'Yes'])
fig.update_traces(textposition = 'inside', 
                  textinfo = 'percent + label', 
                  hole = 0.6, 
                  marker = dict(colors = ['#C2A7B5','#DE1A82'],
                                line = dict(color = 'white', width = 3)))

fig.update_layout(annotations = [dict(text = 'Touch Ekran', 
                                      x = 0.5, y = 0.5,
                                      font_size = 24, showarrow = False, 
                                      font_family = 'Verdana',
                                      font_color = 'black')],
                  showlegend = False)
                  
fig.show()

#############   WİFİ ###############################
fig = px.pie(data['wifi'].value_counts().reset_index(), values = 'wifi',
             names = ['No', 'Yes'])
fig.update_traces(textposition = 'inside', 
                  textinfo = 'percent + label', 
                  hole = 0.6, 
                  marker = dict(colors = ['#C2A7B5','#DE1A82'],
                                line = dict(color = 'white', width = 3)))

fig.update_layout(annotations = [dict(text = 'WİFİ', 
                                      x = 0.5, y = 0.5,
                                      font_size = 24, showarrow = False, 
                                      font_family = 'Verdana',
                                      font_color = 'black')],
                  showlegend = False)
                  
fig.show()

