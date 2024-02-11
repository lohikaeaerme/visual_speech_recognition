def remove_extra_colums(src_path,dest_path):
    colums_count = 3
    with open(src_path) as file:
        with open(dest_path,'x') as new_file:
            for line in file:
                colums = line.split(',')
                new_line = colums[0]
                for i in range(1,colums_count):
                    new_line += ',' + colums[i]
                
                for i in range(colums_count,len(colums)):
                    new_line += ' ' + colums[i]

                new_file.write(new_line)

#remove_extra_colums('make_bundestag_dataset/bundestag_videos/147_unchanged.csv','make_bundestag_dataset/bundestag_videos/147.csv')
#remove_extra_colums('make_bundestag_dataset/bundestag_videos/148_unchanged.csv','make_bundestag_dataset/bundestag_videos/148.csv')
#remove_extra_colums('make_bundestag_dataset/bundestag_videos/149_unchanged.csv','make_bundestag_dataset/bundestag_videos/149.csv')