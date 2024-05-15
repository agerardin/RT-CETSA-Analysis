# #  @@@@@@@   @@@@@@@              @@@@@@@  @@@@@@@@  @@@@@@@   @@@@@@    @@@@@@   
# #  @@@@@@@@  @@@@@@@             @@@@@@@@  @@@@@@@@  @@@@@@@  @@@@@@@   @@@@@@@@  
# #  @@!  @@@    @@!               !@@       @@!         @@!    !@@       @@!  @@@  
# #  !@!  @!@    !@!               !@!       !@!         !@!    !@!       !@!  @!@  
# #  @!@!!@!     @!!    @!@!@!@!@  !@!       @!!!:!      @!!    !!@@!!    @!@!@!@!  
# #  !!@!@!      !!!    !!!@!@!!!  !!!       !!!!!:      !!!     !!@!!!   !!!@!!!!  
# #  !!: :!!     !!:               :!!       !!:         !!:         !:!  !!:  !!!  
# #  :!:  !:!    :!:               :!:       :!:         :!:        !:!   :!:  !:!  
# #  ::   :::     ::                ::: :::   :: ::::     ::    :::: ::   ::   :::  
# # NonParametric Multiparameter Analysis of CETSA/RT-CETSA Experimental Sets
# #
# # Written by: Michael Ronzetti {NIH/NCATS/UMD}
# # Patents: PCT/US21/45184, HHS E-022-2022-0-US-01
# # Main Analysis

suppressWarnings(library(logging))
suppressWarnings(library(tidyverse))
suppressWarnings(library(readxl))
suppressWarnings(library(stringr))
suppressWarnings(library(drc))
suppressWarnings(library(ggthemes))
suppressWarnings(library(cowplot))
suppressWarnings(library(hrbrthemes))
suppressWarnings(library(ggpubr))
suppressWarnings(library(MESS))
suppressWarnings(library(devtools))

loginfo('loading moltenprot fit params from : %s', params)
loginfo('loading moltenprot baseline corrected from : %s', values)
loginfo('loading platemap fit params from : %s', plate_map)

source('./prepare_params.R')

# full_param <- read_csv(params,
# show_col_types = FALSE
# )

source('./prepare_values.R')

# curve_df <- read_csv(values,
# show_col_types = FALSE
# )

platemap_filepath = plate_map

# Assign compound ids and concentration from platemap
plate_assignment <- function(df, platemap_file) {
  # read sample sheet from plate file
  id_df <- read_excel(platemap_file, sheet = 'sample') %>%
    # remove first column
    dplyr::select(-1) %>%
    # pivot to get row, col coordinates as columns
    pivot_longer(cols = 1:ncol(.))  %>%
    # rename the columns with all ids
    rename(ncgc_id = value) %>%
    # remove name column
    dplyr::select(-c('name'))
    # NOTE `EMPTY` are considered as vehicle
    id_df$ncgc_id <- gsub('empty', 'vehicle', id_df$ncgc_id)

    # read the concentration from the file
    conc_df <- read_excel(platemap_file, sheet = 'conc') %>%
    # remove first colum
    dplyr::select(-1) %>%
    # pivot
    pivot_longer(., cols = 1:ncol(.)) %>%
    rename(conc = value) %>%
    dplyr::select(-c('name'))

    # add the columns to the datset
    df <- cbind(id_df, conc_df, df)
  message('Plate assignment attached to dataframe.')

  # make sure we have numeric value? (unecessary?)
  df$row <- as.numeric(df$row)
  df$col <- as.numeric(df$col)
  return(df)
}


full_df <- full_param

full_df <- plate_assignment(full_df, platemap_filepath)

# Construct full data frame with curve fit and parameters for analysis
bind_fulldf <- function(param_df, curve_df) {
  df <- cbind(param_df, curve_df)
  return(df)
}

# Concat dataframes.
full_df <- bind_fulldf(full_df, curve_df)

#Convert any columns containing Kelvin values from MoltenProt to Celsius
kelToCel <- function(df) {
  df <- df %>%
    mutate(Tm_fit = Tm_fit - 273.15) %>%
    mutate(T_onset = T_onset - 273.15)
}

# TODO move that before for each dataset
full_df <- full_df %>% dplyr::select(-c('...1')) %>% dplyr::select(-c('...1')) 
full_df <- kelToCel(full_df)

merge_molten_filename <- "test_merged_molten_final.csv"
output_path <- paste(outdir, merge_molten_filename, sep="/")
loginfo('writing results to : %s', output_path)

write.csv(full_df,output_path)





