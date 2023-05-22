#%%
import altair


#%%
# 1840 word count
alt.Chart(data_journals_grouped_1840, title = 'Wilford Woodruff year 1840').encode(
    x = 'month',
    y = 'word_count'
).mark_line()
#%%
alt.Chart(data_journals_grouped_1840, title = 'Wilford Woodruff year 1847').encode(
    x = 'month',
    y = 'word_count'
).mark_line()
#%%
# 1847 word count
data_journals_grouped_1847 = (data_journals
                         .query('year == 1847')
                         .groupby('month').agg(sum).reset_index())
data_journals_grouped_1847

#%%
# by month
alt.Chart(data_journals_grouped_1847, title = 'Wilford Woodruff year 1847').encode(
    x = 'month',
    y = 'word_count'
).mark_line()
#%%
# by year
alt.Chart(data_journals.query('year == 1847 & month==4'), title = 'Wilford Woodruff year 1847').encode(
    x = 'date',
    y = 'word_count'
).mark_line()

