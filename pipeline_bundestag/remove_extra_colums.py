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
                
                new_line = new_line.replace("/", " ")
                new_file.write(new_line)

#remove_extra_colums('make_bundestag_dataset/bundestag_videos/150_unchanged.csv','make_bundestag_dataset/bundestag_videos/150.csv')
remove_extra_colums('make_bundestag_dataset/bundestag_videos/151_unchanged.csv','make_bundestag_dataset/bundestag_videos/151.csv')
remove_extra_colums('make_bundestag_dataset/bundestag_videos/152_unchanged.csv','make_bundestag_dataset/bundestag_videos/152.csv')
remove_extra_colums('make_bundestag_dataset/bundestag_videos/153_unchanged.csv','make_bundestag_dataset/bundestag_videos/153.csv')
remove_extra_colums('make_bundestag_dataset/bundestag_videos/154_unchanged.csv','make_bundestag_dataset/bundestag_videos/154.csv')