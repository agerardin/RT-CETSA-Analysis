
# values = "/Users/antoinegerardin/RT-CETSA-Analysis/.data/final_outputs/moltenprot/plate_(1-59)_moltenprot_curves.csv"

suppressWarnings(library(logging))
library(tidyverse)

loginfo('loading moltenprot values from : %s', values)

# NOTE this process creates spurious columns that should be removed
exp_curve_all <- read_csv(values,
show_col_types = FALSE
)

# same, row, col grid.
col_by_row <-
  expand.grid(row = sprintf('%.2d', 1:16), col = sprintf('%.2d', 1:24)) %>%
  arrange(., row)


# rename to Temperature
exp_curve_all <- exp_curve_all %>%
# add prefix
mutate(., Temperature = paste('val_t_', Temperature, sep = ''))

exp_curve_all <- exp_curve_all %>%
  # pivot transform columns into row combinations (vy creating a column name)
  pivot_longer(cols = 2:ncol(exp_curve_all)) %>%
  # pivoting again to get temperature as columns
  pivot_wider(names_from = Temperature) %>%
  # create a id column called well 
  rownames_to_column() %>% rename('well' = 'rowname') %>%
  # add the grid coordinates
  bind_cols(col_by_row) %>%
  # remove all unused cols
  dplyr::select(-c('name', 'well', 'row', 'col'))


# Add temperature headers to df
# TODO REVIEW this is sketchy 
add_tempheaders <- function(df,
                            start_temp = 37,
                            end_temp = 90) {
  # generate temperature intervals                       
  temperature_df <-
    seq(start_temp, end_temp, by = ((end_temp - start_temp) / (ncol(df) - 1))) %>%
    round(., digits = 1)
  # rewrite all temperatures!
  # TODO CHECK that, that's quite sketchy. Should convert existing temp
  for (i in 1:ncol(df)) {
    colnames(df)[i] <- paste('t_', temperature_df[i], sep = '')
  }
  message('Temperature assignments changed for ',
          ncol(df),
          ' points.')
  return(df)
}

start_temp = 37
end_temp = 90
# exp_curve_all <- add_tempheaders(exp_curve_all, start_temp, end_temp)
curve_df <- add_tempheaders(exp_curve_all, start_temp, end_temp)
message('Fit curves retrieved.')

# write.csv(exp_curve_all, "test_exp_curve_all.csv")

