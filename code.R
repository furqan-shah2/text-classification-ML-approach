## Load and install libraries
library(dplyr)
library(tidyr)
library(stringr)
library(tidyverse)
library(tidytext)
library(caret)
library(tm)

path <- "F:/Github Projects/text-classification-ML-approach/data files" #Note: Replace the path variable with the path to your own data directory

file_list <- list.files(path)
file_list

read_file <- function(file_name) {
  read_tsv(file.path(path, file_name),
           col_names = TRUE,
           col_types = "cc") %>%
    mutate(file_name = file_name)
}

data <- bind_rows(lapply(file_list, read_file))
data

## setting up and cleaning annotations (training) data ----
annotations <- data %>%
  na.omit() %>%
  rowid_to_column(., "id") %>%
  rename(text = Text,
         theme = Theme)

glimpse(annotations)

##
annotations <- annotations %>%
  mutate_at(vars(text), funs(gsub("[?âè.ã'’'½_&$,%'-']", " ", .))) %>%   # removing special characters
  mutate_at(vars(text), funs(gsub("([0-9])([a-z])", "\\1 \\2", .))) %>%  # creating gap between words and numbers
  mutate_at(vars(text), funs(gsub("([a-z])([0-9])", "\\1 \\2", .))) %>%  # same as above with reversed order
  mutate_at(vars(text), funs(gsub("\\s+", " ", .))) %>%                  # stripping extra white space
  mutate(length = lengths(gregexpr("\\W+", text)) + 1,                   # number of words in an annotation
         mean_length = mean(as.numeric(length))) %>%                     # average length of annotations
  filter(length > 1)                                                     # Filter out short annotations

glimpse(annotations)

annotations %>% count(theme)

## Tokenize annotated data
tidy_annotations <- annotations %>%
  filter(!str_detect(text, "^[0-9]*$")) %>%
  unnest_tokens(word, text) %>%
  anti_join(stop_words) %>%               # removing stop words          
  mutate(characters = nchar(word)) %>%    # number of alphabets in a word
  filter(characters > 3)              # filtering for words with > 3 alphabets

glimpse(tidy_annotations)

## top 15 words - annotated data
top15_tidy <- tidy_annotations %>%
  filter(!str_detect(word, "^[0-9]*$")) %>%
  count(word, theme) %>%
  group_by(theme) %>%
  arrange(desc(n)) %>%
  top_n(15) %>%
  ungroup() %>%
  arrange(theme, word, desc(n))
top15_tidy

top15_tidy_plot <- top15_tidy %>%
  mutate(word = reorder(word, n)) %>%
  ggplot(aes(word, n)) +
  geom_col () +
  labs(x = NULL, y = "n") +
  facet_wrap(~theme, ncol = 2, scales = "free") +
  coord_flip()
top15_tidy_plot


## Create Training and Testing data set ---
set.seed(1234)
train_row_numbers <- createDataPartition(annotations$theme, p = 0.9, list = FALSE)
train_data <- annotations[train_row_numbers, ]
test_data <- annotations[-train_row_numbers, ]


## Setting up train and test data
tidy_train_data <- train_data %>%
  filter(!str_detect(text, "^[0-9]*$")) %>%
  unnest_tokens(word, text) %>%
  anti_join(stop_words) %>%               # removing stop words          
  mutate(characters = nchar(word)) %>%    # number of alphabets in a word
  filter(characters > 3) 

glimpse(tidy_train_data)

tidy_test_data <- test_data %>%
  filter(!str_detect(text, "^[0-9]*$")) %>%
  unnest_tokens(word, text) %>%
  anti_join(stop_words) %>%               # removing stop words          
  mutate(characters = nchar(word)) %>%    # number of alphabets in a word
  filter(characters > 3) 

glimpse(tidy_test_data)

# Creating DTM for train and test data
dtm_train_data <- tidy_train_data %>%
  filter(grepl("^[[:alpha:]]+$", word)) %>%
  count(id, word) %>%
  cast_dtm(document = id,
           term = word,
           value = n,
           weighting = tm::weightTfIdf) %>%
  removeSparseTerms(sparse = 0.99)

dtm_train_data

dtm_test_data <- tidy_test_data %>%
  filter(grepl("^[[:alpha:]]+$", word)) %>%
  count(id, word) %>%
  cast_dtm(document = id,
           term = word,
           value = n,
           weighting = tm::weightTfIdf) %>%
  removeSparseTerms(sparse = 0.997)

dtm_test_data


## Train a random forest model

# slice training data so that only documents in the DTM remain (some were dropped due to sparsity when creating DTM)

doc_names <- as.numeric(dtm_train_data$dimnames$Docs) # Save doc names

slice_train_data <- annotations %>% 
  filter(id %in% doc_names)

glimpse(slice_train_data)

set.seed(1234)

ranger_model <- train(x = as.matrix(dtm_train_data),
                    y = factor(slice_train_data$theme),
                    method = "ranger",
                    importance = "impurity",
                    num.trees = 20,
                    trControl = trainControl(method = "oob"))

ranger_model$finalModel$confusion.matrix

## Implemented the model on the testing data set
predictions_test_data <- predict.train(ranger_model, as.matrix(dtm_test_data))

test_data_slice <- slice(annotations, as.numeric(dtm_test_data$dimnames$Docs)) %>%
  mutate(theme = factor(theme, levels = unique(theme)))

confusion_matrix <- confusionMatrix(reference = test_data_slice$theme, data = predictions_test_data, mode = 'everything')
confusion_matrix


prediction_results <- cbind(test_data_slice, predictions_test_data)
glimpse(prediction_results)

# Plot top 15 words - Classified data
predicted_tidy_top15 <- prediction_results %>%
  unnest_tokens(word, text) %>%
  mutate_all(funs(gsub("([0-9])([a-z])", "\\1 \\2", .))) %>%
  mutate_all(funs(gsub("([a-z])([0-9])", "\\1 \\2", .))) %>%
  unnest_tokens(word, word) %>%
  mutate_all(funs(gsub("[?âè.''_&$,%]"," ", .))) %>%
  unnest_tokens(word, word) %>%
  anti_join(stop_words) %>%
  filter(!str_detect(word, "^[0-9]*$")) %>%
  count(word, predictions_test_data) %>%
  group_by(predictions_test_data) %>%
  arrange(desc(n)) %>%
  top_n(15) %>%
  ungroup() %>%
  arrange(predictions_test_data, desc(n))
predicted_tidy_top15

predicted_top15_plot <- 
  predicted_tidy_top15 %>%
  mutate(word = reorder(word, n)) %>%
  ggplot(aes(word, n)) +
  geom_col () +
  labs(x = NULL, y = "n", title = "Top 15 Words - Classified") +
  facet_wrap(~predictions_test_data, ncol = 2, scales = "free") +
  coord_flip()
predicted_top15_plot

## Plot top 15 words - training data
train_tidy_top15 <- train_data %>%
  unnest_tokens(word, text) %>%
  mutate_all(funs(gsub("([0-9])([a-z])", "\\1 \\2", .))) %>%
  mutate_all(funs(gsub("([a-z])([0-9])", "\\1 \\2", .))) %>%
  unnest_tokens(word, word) %>%
  mutate_all(funs(gsub("[?âè.''_&$,%]"," ", .))) %>%
  unnest_tokens(word, word) %>%
  anti_join(stop_words) %>%
  filter(!str_detect(word, "^[0-9]*$")) %>%
  count(word, theme) %>%
  group_by(theme) %>%
  arrange(desc(n)) %>%
  top_n(15) %>%
  ungroup() %>%
  arrange(theme, desc(n))
train_tidy_top15


train_tidy_top15_plot <- train_tidy_top15 %>%
  mutate(word = reorder(word, n)) %>%
  ggplot(aes(word, n)) +
  geom_col () +
  labs(x = NULL, y = "n", title = "Top 15 Words - Training") +
  facet_wrap(~theme, ncol = 2, scales = "free") +
  coord_flip()
train_tidy_top15_plot
