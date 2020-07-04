%this matlab script is for compute PESQ, SSNR.
%reference: https://ecs.utdallas.edu/loizou/speech/software.htm

function measure_Csig_Cbak_Covl_segSNR_pesq_composite(clean_root, enhan_root, distorted_root, scores_file_path)
addpath('../../../metrics/composite'); 

scores = [];
fid = fopen(scores_file_path,'w');
fprintf(fid,'item, Csig_enhan, Cbak_enhan, Covl_enhan, segSNR_enhan, pesq_mos_enhan, Csig_d, Cbak_d, Covl_d, segSNR_d, pesq_mos_d\n');


clean_file_list = dir(clean_root);
clean_name_list = {clean_file_list(3:end).name};

%FS = 16000;
Csig_enhan_sum = 0;
Cbak_enhan_sum = 0;
Covl_enhan_sum = 0;
segSNR_enhan_sum = 0;
pesq_enhan_sum = 0;

Csig_d_sum = 0;
Cbak_d_sum = 0;
Covl_d_sum = 0;
segSNR_d_sum = 0;
pesq_d_sum = 0;

for i = 1:length(clean_name_list)
    item = clean_name_list(i);
    item = item{1};
    clean_file = [clean_root item(1:end-4) '.wav'];
    
    enhanced_file = [enhan_root item(1:end-4) '_enhan.wav'];
    
    distorted_file = [distorted_root item(1:end-4) '.wav'];
    
    [Csig_enhan,Cbak_enhan,Covl_enhan, llr_mean_enhan, segSNR_enhan, wss_dist_enhan,pesq_mos_enhan]=composite(clean_file,enhanced_file);
    
    [Csig_d,Cbak_d,Covl_d, llr_mean_d, segSNR_d, wss_dist_d,pesq_mos_d]=composite(clean_file,distorted_file);
    
    
    Csig_enhan_sum = Csig_enhan_sum + Csig_enhan;
    Cbak_enhan_sum = Cbak_enhan_sum + Cbak_enhan;
    Covl_enhan_sum = Covl_enhan_sum + Covl_enhan;
    segSNR_enhan_sum = segSNR_enhan_sum + segSNR_enhan;
    pesq_enhan_sum = pesq_enhan_sum + pesq_mos_enhan;
    
    Csig_d_sum = Csig_d_sum + Csig_d;
    Cbak_d_sum = Cbak_d_sum + Cbak_d;
    Covl_d_sum = Covl_d_sum + Covl_d;
    segSNR_d_sum = segSNR_d_sum + segSNR_d;
    pesq_d_sum = pesq_d_sum + pesq_mos_d;
    
    scores{end+1} = {item Csig_enhan Cbak_enhan Covl_enhan segSNR_enhan pesq_mos_enhan Csig_d Cbak_d Covl_d segSNR_d pesq_mos_d};
end

avg_Csig_enhan = Csig_enhan_sum / i;
avg_Cbak_enhan = Cbak_enhan_sum / i;
avg_Covl_enhan = Covl_enhan_sum / i;
avg_segSNR_enhan = segSNR_enhan_sum / i;
avg_pesq_enhan = pesq_enhan_sum / i;

avg_Csig_d = Csig_d_sum / i;
avg_Cbak_d = Cbak_d_sum / i;
avg_Covl_d = Covl_d_sum / i;
avg_segSNR_d = segSNR_d_sum / i;
avg_pesq_d = pesq_d_sum / i;

scores{end+1} = {'average:' avg_Csig_enhan avg_Cbak_enhan avg_Covl_enhan avg_segSNR_enhan avg_pesq_enhan avg_Csig_d avg_Cbak_d avg_Covl_d avg_segSNR_d avg_pesq_d};


for j= 1:length(scores)
    fprintf(fid,'%s %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f\n', scores{j}{:});
end

fclose(fid);

end
