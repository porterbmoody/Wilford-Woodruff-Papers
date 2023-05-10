pacman::p_load(tidyverse, ggrepel)
theme_set(theme_bw())
getwd()

journal_entries = 'data/wwp_journal_entries_from1836_to1895.csv'
raw_entries = 'data/raw_entries.csv'
raw_entries_clean = 'data/raw_entries_clean.csv'

data_journals <- read_csv(raw_entries_clean)
data_journals %>% view
data_journals

data1 <- data_journals %>% 
  mutate(word_count = lengths(gregexpr("\\W+", text)) + 1,
         year = mdy(date) %>% year(),
         month = mdy(date) %>% month(),
         # joseph_mentions = sum(str_count(dataset, regex("joseph smith")))
         )
data1 %>% write_csv('data/raw_entries_clean.csv')

data_grouped <- data1 %>% 
  group_by(year) %>% 
  summarise(total_word_count = sum(word_count))
data_grouped$year


# total word count by year
data_grouped %>% 
  ggplot() + 
  aes(x = year, 
      y = total_word_count) + 
  geom_line() +
  geom_point() +
  labs(title = 'Wilford Woodruff journal entry total word count by year',
       y = 'Word Count') +
  scale_x_continuous(breaks = c(data_grouped$year)) +
  scale_y_continuous(breaks = c(seq(from = 10000, to = 100000, by = 10000))) +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) +
  geom_label_repel(data=filter(data_grouped,total_word_count>40000),
                   aes(label = paste0(year, ': ', total_word_count, ' words')), nudge_x = 1, size =3)


data_grouped %>% 
  ggplot() +
  aes()





