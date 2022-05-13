from gaps_dataset import gaps
gaps.download(login='gapsro2s;i2A*7',
                    output_dir='datasets',
                    version=2,
                    patchsize=160,
                    issue='NORMvsDISTRESS_50k',
                    debug_outputs=True)

gaps.download(login='gapsro3Z=1Yb%7',
                    output_dir='datasets',
                    version='10m',
                    patchsize='segmentation',
                    issue='ASFaLT',
                    debug_outputs=True)


# x_train0, y_train0 = gaps.load_chunk(chunk_id=0,
#                                     version=2,
#                                     patchsize=160,
#                                     issue='NORMvsDISTRESS_50k',
#                                     subset='train',
#                                     datadir='desired folder (same folder used in download function)') #load the first chunk of the dataset