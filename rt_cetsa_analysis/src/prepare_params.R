suppressWarnings(library(logging))
library(tidyverse)

# params = "/Users/antoinegerardin/RT-CETSA-Analysis/.data/final_outputs/moltenprot/plate_(1-59)_moltenprot_params.csv"

loginfo('loading moltenprot params from : %s', params)

# create a dataframe with two columns (row, col) for plate of (16,24)
col_by_row <- expand.grid(row = sprintf('%.2d', 1:16), col = sprintf('%.2d', 1:24)) %>% 
# sort by row number
arrange(., row)

# NOTE this process creates spurious columns that should be removed
exp_param <- read_csv(params,
show_col_types = FALSE
)

# create a col with sequential ids for indexing
# named it `well`
exp_param <- exp_param %>% rownames_to_column() %>% rename('well' = 'rowname')

# select columns we need for this analysis
exp_param <- exp_param %>%
        dplyr::select(
          c(
            'well',
            'dHm_fit',
            'Tm_fit',
            'BS_factor',
            'T_onset',
            'dG_std'
        ))

# add row col info to results (after the well column)
exp_param <- exp_param %>% bind_cols(col_by_row) %>% relocate(c('row', 'col'), .after = well) 

# NOTE why do we create well at the first place?
# remove the well column
exp_param <- exp_param %>% dplyr::select(-'well')

# NOTE Basically regenerate the battleship coordinates based on the current ordering
# Add well assignments for each plate
well_assignment <- function(df, well_num) {
  if (well_num == 96) {
    letter <- LETTERS[1:8]
    number <- c(1:12)
    number <- str_pad(number, 2, pad = '0')
    tracker <- 1
    temp_df <- tibble(well = c(1:384))
    for (val in letter) {
      for (num in number) {
        temp_df$well[tracker] <- paste(val, num, sep = '')
        tracker <- tracker + 1
      }
    }
  }
  else if (well_num == 384) {
    letter <- LETTERS[1:16]
    number <- c(1:24)
    number <- str_pad(number, 2, pad = '0')
    tracker <- 1
    temp_df <- tibble(well = c(1:384))
    for (val in letter) {
      for (num in number) {
        temp_df$well[tracker] <- paste(val, num, sep = '')
        tracker <- tracker + 1
      }
    }
  }
  message('Well assignments created for ', well_num, '-well plate.')
  df <- cbind(temp_df, df)
  return(df)
}

full_param <- well_assignment(exp_param, 384)

# write.csv(exp_param, "test_exp_param_full.csv")