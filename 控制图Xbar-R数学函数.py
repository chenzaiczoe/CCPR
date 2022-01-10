
import numpy as np
import pandas as pd
import random
import plotly.offline as py
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# -------------数据导入----------------- #
df = pd.read_csv('D:\\holley\\pdtC.csv')

# ——————数据整理————————#
df['time'] = pd.to_datetime(df['time'],
                            format='%Y/%m/%d %H:%M')

df1 = df.copy()
df1['counts'] = 0
df2 = df1.groupby(['time', 'IP'])['counts'].count().reset_index()
result = pd.merge(df, df2, how='left', on=['time', 'IP'])  # 相同时间及IP为一组

# 6858 rows x 6 columns]
result = result.drop(result[(result.counts < 5)].index)  # 删除子项少于5的组

result.sort_values('time', inplace=True)


np.random.seed(9)
group_sp = []
for (time, IP), group in result.groupby(['time', 'IP']):
    group_sp.append(random.sample(list(group.w), 5))
data = np.array(group_sp).reshape(470, 5)

# 8 Rules ------------------------------------------------------


def rules(data, cl, ucl, ucl_b, ucl_c, lcl, lcl_b, lcl_c):
    n = len(data)
    ind = np.array(range(n))
    obs = np.arange(1, n + 1)

    # rule 1 界外
    ofc1 = data[(data > ucl) | (data < lcl)]
    ofc1_obs = obs[(data > ucl) | (data < lcl)]

    # rule 2 连续3点中有2点落在中心线同一侧的B区以外
    ofc2_ind = []
    for i in range(n - 2):
        d = data[i:i + 3]
        index = ind[i:i + 3]
        if ((d > ucl_b).sum() == 2) | ((d < lcl_b).sum() == 2):
            ofc2_ind.extend(index[(d > ucl_b) | (d < lcl_b)])
    ofc2_ind = list(sorted(set(ofc2_ind)))
    ofc2 = data[ofc2_ind]
    ofc2_obs = obs[ofc2_ind]

    # rule 3 连续5点中有4点落在中心线同一侧的C区以外
    ofc3_ind = []
    for i in range(n - 4):
        d = data[i:i + 5]
        index = ind[i:i + 5]
        if ((d > ucl_c).sum() == 4) | ((d < lcl_c).sum() == 4):
            ofc3_ind.extend(index[(d > ucl_c) | (d < lcl_c)])
    ofc3_ind = list(sorted(set(ofc3_ind)))
    ofc3 = data[ofc3_ind]
    ofc3_obs = obs[ofc3_ind]

    # rule 4 连续9点落在中心线同一侧
    ofc4_ind = []
    for i in range(n - 8):
        d = data[i:i + 9]
        index = ind[i:i + 9]
        if ((d > cl).sum() == 9) | ((d < cl).sum() == 9):
            ofc4_ind.extend(index)
    ofc4_ind = list(sorted(set(ofc4_ind)))
    ofc4 = data[ofc4_ind]
    ofc4_obs = obs[ofc4_ind]

    # rule 5 连续6点递增或递减
    ofc5_ind = []
    for i in range(n - 6):
        d = data[i:i + 7]
        index = ind[i:i + 7]
        if all(u <= v for u, v in zip(d, d[1:])) | all(
                u >= v for u, v in zip(d, d[1:])):
            ofc5_ind.extend(index)
    ofc5_ind = list(sorted(set(ofc5_ind)))
    ofc5 = data[ofc5_ind]
    ofc5_obs = obs[ofc5_ind]

    # rule 6 连续8点在中心线两侧，但无一点在C区中
    ofc6_ind = []
    for i in range(n - 7):
        d = data[i:i + 8]
        index = ind[i:i + 8]
        if (all(d > ucl_c) | all(d < lcl_c)):
            ofc6_ind.extend(index)
    ofc6_ind = list(sorted(set(ofc6_ind)))
    ofc6 = data[ofc6_ind]
    ofc6_obs = obs[ofc6_ind]

    # rule 7 	连续15点在C区中心线上下
    ofc7_ind = []
    for i in range(n - 14):
        d = data[i:i + 15]
        index = ind[i:i + 15]
        if all(lcl_c < d) and all(d < ucl_c):
            ofc7_ind.extend(index)
    ofc7_ind = list(sorted(set(ofc7_ind)))
    ofc7 = data[ofc7_ind]
    ofc7_obs = obs[ofc7_ind]

    # rule 8 连续14中相邻点上下交替
    ofc8_ind = []
    for i in range(n - 13):
        d = data[i:i + 14]
        index = ind[i:i + 14]
        diff = list(v - u for u, v in zip(d, d[1:]))
        if all(u * v < 0 for u, v in zip(diff, diff[1:])):
            ofc8_ind.extend(index)
    ofc8_ind = list(sorted(set(ofc8_ind)))
    ofc8 = data[ofc8_ind]
    ofc8_obs = obs[ofc8_ind]

    return ofc1, ofc1_obs, ofc2, ofc2_obs, ofc3, ofc3_obs, ofc4, ofc4_obs, ofc5, ofc5_obs, ofc6, ofc6_obs, ofc7, ofc7_obs, ofc8, ofc8_obs


R = [np.max(i) - np.min(i) for i in group_sp]
X_bar = [np.mean(i) for i in data]
X_arr_std = [np.std(i, ddof=1) for i in data]

k = 0
h = 0
dict_outside = []
for i in range(0, len(data) + 24, 25):
    # -------R chart------#
    R_bar = np.mean(R[int(i):int(i + 25)])
    RUCL = float(R_bar * 2.114)
    RUCL_b = float(R_bar * 2.114 * (5 / 6))
    RUCL_c = float(R_bar * 2.114 * (4 / 6))
    RLCL_c = float(R_bar * 2.114 * (2 / 6))
    RLCL_b = float(R_bar * 2.114 * (1 / 6))
    RLCL = float(0)
    # ---------X bar chart--------#
    x_bar = np.mean(X_bar[int(i):int(i + 25)])
    xUCL = float(x_bar + R_bar * 0.577)
    xUCL_b = float(x_bar + R_bar * 0.577 * (2 / 3))
    xUCL_c = float(x_bar + R_bar * 0.577 * (1 / 3))
    xLCL_c = float(x_bar - R_bar * 0.577 * (1 / 3))
    xLCL_b = float(x_bar - R_bar * 0.577 * (2 / 3))
    xLCL = float(x_bar - R_bar * 0.577)

    # # R Mask
    R_arr = np.array(R[int(i):int(i + 25)])
    _, ind1, _, ind2, _, ind3, _, ind4, _, ind5, _, ind6, _, ind7, _, ind8 \
        = rules(R_arr, R_bar, RUCL, RUCL_b, RUCL_c, RLCL, RLCL_b, RLCL_c)
    ind_R = list(
        set(ind1).union(
            set(ind2)).union(
            set(ind3)).union(
                set(ind4)).union(
                    set(ind5)).union(
                        set(ind6)).union(
                            set(ind7)).union(
                                set(ind8)))
    mask_R = [False]
    for i in range(25):
        if i + 1 in ind_R:
            mask_R.append(True)
        else:
            mask_R.append(False)
    fig = make_subplots(specs=[[{'secondary_y': True}]])
    colors_2 = ['RoyalBlue' if x == False else 'crimson' for x in mask_R]
    fig.add_trace(go.Scatter(x=np.arange(25), y=R[int(i):int(i + 25)],
                             mode='lines+markers',
                             line_color='RoyalBlue',
                             marker_color=colors_2,
                             line=dict(width=1),
                             marker=dict(size=5),
                             name='R'),
                  secondary_y=False)
    fig.update_layout(hovermode='x',
                      title='R chart',
                      showlegend=False)
    fig.update_xaxes(title='Sample',
                     tick0=0, dtick=10,
                     ticks='outside', tickwidth=1, tickcolor='black',
                     range=[0, 25],
                     zeroline=False,
                     showgrid=False)
    fig.update_yaxes(title='R',
                     ticks='outside', tickwidth=1, tickcolor='black',
                     range=[RLCL, RUCL + RUCL * 0.1],
                     nticks=5,
                     showgrid=False,
                     secondary_y=False)
    fig.add_shape(
        type='line',
        line_color='crimson',
        line_width=1,
        x0=0,
        x1=25,
        xref='x1',
        y0=RUCL,
        y1=RUCL,
        yref='y2',
        secondary_y=True)
    fig.add_shape(
        type='line',
        line_color='LightSeaGreen',
        line_dash='dot',
        line_width=1,
        x0=0,
        x1=25,
        xref='x1',
        y0=RUCL_b,
        y1=RUCL_b,
        yref='y2',
        secondary_y=True)
    fig.add_shape(
        type='line',
        line_color='LightSeaGreen',
        line_dash='dot',
        line_width=1,
        x0=0,
        x1=25,
        xref='x1',
        y0=RUCL_c,
        y1=RUCL_c,
        yref='y2',
        secondary_y=True)
    fig.add_shape(
        type='line',
        line_color='LightSeaGreen',
        line_width=1,
        x0=0,
        x1=25,
        xref='x1',
        y0=R_bar,
        y1=R_bar,
        yref='y2',
        secondary_y=True)
    fig.add_shape(
        type='line',
        line_color='LightSeaGreen',
        line_dash='dot',
        line_width=1,
        x0=0,
        x1=25,
        xref='x1',
        y0=RLCL_c,
        y1=RLCL_c,
        yref='y2',
        secondary_y=True)
    fig.add_shape(
        type='line',
        line_color='crimson',
        line_width=1,
        x0=0,
        x1=25,
        xref='x1',
        y0=RLCL,
        y1=RLCL,
        yref='y2',
        secondary_y=True)
    fig.update_yaxes(ticks='outside', tickwidth=1, tickcolor='black',
                     range=[RLCL, RUCL + RUCL * 0.01],
                     ticktext=['LCL=' + str(np.round(RLCL, 3)),
                               'R-bar=' + str(np.round(R_bar, 3)),
                               'UCL=' + str(np.round(RUCL, 3))],
                     tickvals=[RLCL, R_bar, RUCL],
                     showgrid=False,
                     secondary_y=True)

    py.plot(fig, filename='R chart.html')

   # ##x bar chart
    # -----------布尔值的列表遮罩，为之后异常值标记点颜色格式的套用作准备
    # # CL Mask
    x_arr = np.array(X_bar[int(i):int(i + 25)])
    _, ind1, _, ind2, _, ind3, _, ind4, _, ind5, _, ind6, _, ind7, _, ind8 \
        = rules(x_arr, x_bar, xUCL, xUCL_b, xUCL_c, xLCL, xLCL_b, xLCL_c)
    ind_x = list(
        set(ind1).union(
            set(ind2)).union(
            set(ind3)).union(
                set(ind4)).union(
                    set(ind5)).union(
                        set(ind6)).union(
                            set(ind7)).union(
                                set(ind8)))
    mask_cl = [False]
    for i in range(25):
        if i + 1 in ind_x:
            mask_cl.append(True)
        else:
            mask_cl.append(False)

    # -----
    # 新建带有主副 y 轴的画布
    fig = make_subplots(specs=[[{'secondary_y': True}]])
    # 带条件的颜色列表
    colors_1 = ['RoyalBlue' if x == False else 'crimson' for x in mask_cl]
    # 折线图主体
    fig.add_trace(go.Scatter(x=np.arange(25), y=X_bar[int(i):int(i + 25)],
                             mode='lines+markers',
                             line_color='RoyalBlue',
                             marker_color=colors_1,
                             line=dict(width=1),
                             marker=dict(size=5),
                             name='x'),
                  secondary_y=False)
    # 设置布局
    fig.update_layout(hovermode='x',
                      title='x-bar chart',
                      showlegend=False)
    # 设置 x 轴
    fig.update_xaxes(title='Sample',
                     tick0=0, dtick=10,
                     ticks='outside', tickwidth=1, tickcolor='black',
                     range=[0, 25],
                     zeroline=False,
                     showgrid=False)
    # 设置主 y 轴
    fig.update_yaxes(title='x',
                     ticks='outside', tickwidth=1, tickcolor='black',
                     range=[xLCL - xLCL * 0.02, xUCL + xUCL * 0.02],
                     nticks=5,
                     showgrid=False,
                     secondary_y=False)
    # UCL 辅助线
    fig.add_shape(
        type='line',
        line_color='crimson',
        line_width=1,
        x0=0,
        x1=25,
        xref='x1',
        y0=xUCL,
        y1=xUCL,
        yref='y2',
        secondary_y=True)
    # UCL_b 辅助线
    fig.add_shape(
        type='line',
        line_color='LightSeaGreen',
        line_dash='dot',
        line_width=1,
        x0=0,
        x1=25,
        xref='x1',
        y0=xUCL_b,
        y1=xUCL_b,
        yref='y2',
        secondary_y=True)
    # UCL_c 辅助线
    fig.add_shape(
        type='line',
        line_color='LightSeaGreen',
        line_dash='dot',
        line_width=1,
        x0=0,
        x1=25,
        xref='x1',
        y0=xUCL_c,
        y1=xUCL_c,
        yref='y2',
        secondary_y=True)
    # 均值辅助线
    fig.add_shape(
        type='line',
        line_color='LightSeaGreen',
        line_width=1,
        x0=0,
        x1=25,
        xref='x1',
        y0=x_bar,
        y1=x_bar,
        yref='y2',
        secondary_y=True)
    # LCL_c 辅助线
    fig.add_shape(
        type='line',
        line_color='LightSeaGreen',
        line_dash='dot',
        line_width=1,
        x0=0,
        x1=25,
        xref='x1',
        y0=xLCL_c,
        y1=xLCL_c,
        yref='y2',
        secondary_y=True)
    # LCL_b 辅助线
    fig.add_shape(
        type='line',
        line_color='LightSeaGreen',
        line_dash='dot',
        line_width=1,
        x0=0,
        x1=25,
        xref='x1',
        y0=xLCL_b,
        y1=xLCL_b,
        yref='y2',
        secondary_y=True)
    # LCL 辅助线
    fig.add_shape(
        type='line',
        line_color='crimson',
        line_width=1,
        x0=0,
        x1=25,
        xref='x1',
        y0=xLCL,
        y1=xLCL,
        yref='y2',
        secondary_y=True)
    # 设置副 y 轴 为了方便标记界限值
    fig.update_yaxes(ticks='outside', tickwidth=1, tickcolor='black',
                     range=[xLCL - xLCL * 0.02, xUCL + xUCL * 0.02],
                     ticktext=['LCL=' + str(np.round(xLCL, 3)),
                               'x-bar=' + str(np.round(x_bar, 3)),
                               'UCL=' + str(np.round(xUCL, 3))],
                     tickvals=[xLCL, x_bar, xUCL],
                     showgrid=False,
                     secondary_y=True)

    py.plot(fig, filename='x_bar chart.html')