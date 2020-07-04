%this matlab script is for compute PESQ, SSNR.
%reference: https://ecs.utdallas.edu/loizou/speech/software.htm

function measure_Csig_Cbak_Covl_segSNR_pesq_K14513_CD()
addpath('../K14513_CD_Files/MATLAB_code/objective_measures/quality'); 

%here add your own data path
clean_root=''; 
enhan_root=''; 
scores_file_path='';

scores = [];
fid = fopen(scores_file_path,'w');
fprintf(fid,'item, Csig_enhan, Cbak_enhan, Covl_enhan, segSNR_enhan, pesq_enhan\n');


clean_root_list = dir(clean_root);
clean_name_list = {clean_root_list(3:end).name};

%FS = 16000;
Csig_enhan_sum = 0;
Cbak_enhan_sum = 0;
Covl_enhan_sum = 0;
segSNR_enhan_sum = 0;
pesq_enhan_sum = 0;

for i = 1:length(clean_name_list)
    item = clean_name_list(i);
    item = item{1};
    clean_file = [clean_root item(1:end-4) '.wav'];
    
    enhanced_file = [enhan_root item(1:end-4) '.wav'];
    
    disp(enhanced_file);
    [Csig_enhan,Cbak_enhan,Covl_enhan]=composite(clean_file, enhanced_file);
    pesq_enhan = pesq(clean_file, enhanced_file);
    [snr_enhan, segSNR_enhan]= comp_snr(clean_file, enhanced_file);
%     pesq_enhan = pesq(enhanced_file, distorted_file);
    
    Csig_enhan_sum = Csig_enhan_sum + Csig_enhan;
    Cbak_enhan_sum = Cbak_enhan_sum + Cbak_enhan;
    Covl_enhan_sum = Covl_enhan_sum + Covl_enhan;
    segSNR_enhan_sum = segSNR_enhan_sum + segSNR_enhan;
    pesq_enhan_sum = pesq_enhan_sum + pesq_enhan;
    
    scores{end+1} = {item Csig_enhan Cbak_enhan Covl_enhan segSNR_enhan pesq_enhan};
end

avg_Csig_enhan = Csig_enhan_sum / i;
avg_Cbak_enhan = Cbak_enhan_sum / i;
avg_Covl_enhan = Covl_enhan_sum / i;
avg_segSNR_enhan = segSNR_enhan_sum / i;
avg_pesq_enhan = pesq_enhan_sum / i;


scores{end+1} = {'average:' avg_Csig_enhan avg_Cbak_enhan avg_Covl_enhan avg_segSNR_enhan avg_pesq_enhan};


for j= 1:length(scores)
    fprintf(fid,'%s %.4f %.4f %.4f %.4f %.4f\n', scores{j}{:});
end

fclose(fid);

end
