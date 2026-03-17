try(library(shiny))
try(library(mlr3))
try(library(mlr3learners))
try(library(mlr3pipelines))
# try(library(mlr3extralearners))  # 注释掉：该包不存在
try(library(ggplot2))
try(library(shapviz))
try(library(kernelshap))

# --- 1. 配置名称映射与清洗函数 ---
display_names_map <- c(
  "AST" = "AST (U/L)",
  "PLT" = "PLT (×10⁹/L)", 
  "gender" = "Gender",
  "number.of.metastatic.organs" = "Number of metastatic organs (n)",
  "other.site.metastasis" = "Other site metastasis (n)",
  "primary.tumor.sites" = "Primary tumor site"
)

clean_name <- function(x) {
  if (x %in% names(display_names_map)) {
    return(display_names_map[[x]])
  }
  return(gsub("\\.", " ", x))
}

clean_name_for_plot <- function(x) {
  name <- clean_name(x)
  return(gsub("\\s*\\(.*?\\)", "", name)) 
}

gender_choices <- list("Male" = 0, "Female" = 1)
site_choices <- list("left colon cancer" = 1, "right colon cancer" = 2, "rectal cancer" = 3)

# --- 2. 加载模型逻辑 ---
# 使用 app 所在目录加载 RData，确保部署时路径正确
rdata_path <- file.path(getwd(), "image_ChooseModel.RData")

# 初始化必要的全局变量
model_loaded <- FALSE
task_model   <- NULL
variables    <- list()
train_data   <- data.frame()

if (file.exists(rdata_path)) {
  tryCatch({
    load(rdata_path, envir = .GlobalEnv)
    
    # 策略：优先直接使用已训练好的模型对象（含 $state），无需重新 train
    # 只有当模型尚未训练（$state 为 NULL）时才重新训练 pipeline
    needs_train <- is.null(model_ChooseModel_aftertune$state)
    
    if (needs_train) {
      # 优化参数：移除无效的参数（如 mtry），只保留有效的参数
      if (exists("best_ChooseModel_param_vals") && !is.null(best_ChooseModel_param_vals)) {
        valid_params <- graph_pipeline_ChooseModel$param_set$ids()
        # 过滤掉无效的参数
        valid_param_vals <- best_ChooseModel_param_vals[names(best_ChooseModel_param_vals) %in% valid_params]
        graph_pipeline_ChooseModel$param_set$set_values(.values = valid_param_vals)
      }
      graph_pipeline_ChooseModel$train(task_train)
    }
    
    task_model  <- model_ChooseModel_aftertune$state$train_task
    variables   <- setNames(as.list(task_model$feature_types$type), task_model$feature_types$id)
    train_data  <- as.data.frame(task_train$data())
    model_loaded <- TRUE
  }, error = function(e) {
    warning(paste("模型加载失败：", conditionMessage(e)))
  })
} else {
  warning(paste("找不到模型文件：", rdata_path, "——请确认已将 image_ChooseModel.RData 与 app.R 放在同一目录"))
}

# 定义自定义默认值
custom_defaults <- list(
  "AST" = 20,  # AST默认值设为20
  "PLT" = 239,  # PLT默认值设为239
  "gender" = 1,  # 性别默认设为女性 (1)
  "number.of.metastatic.organs" = 1,  # 转移器官数量默认设为1
  "other.site.metastasis" = 0,  # 其他部位转移默认设为0
  "primary.tumor.sites" = 1  # 原发肿瘤部位默认设为左结肠癌
)

# 计算默认值的函数
get_default_value <- function(feature) {
  if (feature %in% names(custom_defaults)) {
    return(custom_defaults[[feature]])
  } else if (variables[[feature]] %in% c("numeric", "integer")) {
    return(median(train_data[[feature]], na.rm = TRUE))
  } else {
    return(task_model$levels(feature)[[1]][1])
  }
}

# 🚀 优化 1：检查全局SHAP缓存文件，避免重复计算（启动快 90%）
SHAP_cache_path <- file.path(getwd(), "SHAP_sv_ChooseModel.rds")

if (!exists("SHAP_sv_ChooseModel")) {
  # 首先尝试从缓存加载
  if (file.exists(SHAP_cache_path)) {
    tryCatch({
      SHAP_sv_ChooseModel <- readRDS(SHAP_cache_path)
      print(paste("✅ 全局SHAP已从缓存加载 (", file.size(SHAP_cache_path) / 1024 / 1024, "MB )"))
    }, error = function(e) {
      warning(paste("缓存加载失败:", e$message, "——将重新计算"))
    })
  }
  
  # 如果缓存不存在或加载失败，则计算
  if (!exists("SHAP_sv_ChooseModel")) {
    tryCatch({
      # 如果SHAP对象不存在，尝试从数据计算
      library(shapviz)
      library(kernelshap)
      
      print("⏳ 正在计算全局SHAP值... (首次启动需要 10-30 秒)")
      
      # 创建预测函数
      pred_fun <- function(object, newdata) {
        pred <- object$predict_newdata(newdata)
        if ("prob.1" %in% names(as.data.table(pred))) {
          return(as.numeric(as.data.table(pred)$prob.1))
        } else {
          return(as.numeric(as.data.table(pred)$response))
        }
      }
      
      # 使用部分数据计算全局SHAP值（减少计算时间）
      feature_names <- task_model$feature_names
      
      # 🚀 优化 3：减少背景样本数量（从100改为30）
      bg_data <- train_data[sample(nrow(train_data), min(30, nrow(train_data))), feature_names, drop = FALSE]
      
      # 计算kernelshap - 全局SHAP值
      # 注意：kernelshap 使用 m 参数控制蒙特卡洛迭代次数，不支持 n_samples
      # 🚀 优化 2：减少蒙特卡洛迭代次数（从100改为50，可选改为30以获得更快速度）
      shap_values <- kernelshap(
        object = model_ChooseModel_aftertune,
        X = train_data[sample(nrow(train_data), min(300, nrow(train_data))), feature_names, drop = FALSE],
        bg_X = bg_data,
        pred_fun = pred_fun,
        m = 50  # 蒙特卡洛迭代次数（优化：从100改为50，快速且准确）
      )
      
      # 创建shapviz对象
      SHAP_sv_ChooseModel <- shapviz(shap_values)
      
      # 💾 保存缓存，下次启动时直接加载（节省 90% 的启动时间）
      tryCatch({
        saveRDS(SHAP_sv_ChooseModel, SHAP_cache_path)
        print(paste("✅ 全局SHAP已保存到缓存 (", file.size(SHAP_cache_path) / 1024 / 1024, "MB )"))
      }, error = function(e) {
        warning(paste("缓存保存失败:", e$message))
      })
      
      print("✅ 全局SHAP对象已成功计算并创建")
    }, error = function(e) {
      print(paste("❌ 无法计算全局SHAP对象:", e$message))
    })
  }
}

# --- 3. UI 界面布局（包含全局SHAP分析）---
ui <- fluidPage(
  tags$head(
    tags$style(HTML("
      body { 
        background-color: #E6F7FF;
        font-family: 'Segoe UI', Arial, sans-serif; 
        margin: 0;
        padding: 0;
      }
      .navbar-custom { 
        background-color: #2c77b4; 
        color: white; 
        padding: 12px 20px; 
        border-radius: 0 0 10px 10px; 
        margin-bottom: 20px; 
        box-shadow: 0 2px 8px rgba(0,0,0,0.1); 
      }
      .card { 
        background: white; 
        padding: 15px;
        border-radius: 12px; 
        box-shadow: 0 2px 8px rgb(44 119 180); 
        margin-bottom: 15px;
        text-align: left; 
      }
      .section-title { 
        color: #2c77b4; 
        font-weight: bold; 
        border-left: 4px solid #2c77b4; 
        padding-left: 10px; 
        margin-bottom: 12px;
        font-size: 16px;
      }
      
      .risk-container { 
        position: relative; 
        width: 95%;  /* 改为95% */
        height: 16px;
        background: linear-gradient(to right, 
          #5cb85c 0%, #5cb85c 30%, 
          #f0ad4e 30%, #f0ad4e 50%, 
          #d9534f 50%, #d9534f 100%); 
        border-radius: 8px; 
        margin: 12px auto 5px auto;  /* 增加auto使其居中 */
      }
      
      .risk-indicator {
        position: absolute;
        top: -4px;
        width: 2px;
        height: 24px;
        background-color: #1a1a1a;
        border-radius: 1px;
        transform: translateX(-50%);
        transition: left 0.5s ease-out;
      }

      .scale-wrapper { 
        position: relative; 
        width: 95%;  /* 改为95% */
        height: 18px;
        color: #666; 
        font-size: 12px;
        font-weight: 500; 
        margin-bottom: 12px;
        margin-left: auto;  /* 增加自动外边距使其居中 */
        margin-right: auto;  /* 增加自动外边距使其居中 */
      }
      .scale-label { 
        position: absolute; 
        transform: translateX(-50%); 
      }
      
      .risk-label-badge { 
        display: inline-block; 
        padding: 4px 16px;
        border-radius: 20px; 
        color: white; 
        font-weight: bold; 
        font-size: 14px;
        margin-top: 5px;
      }
      .prob-text-style { 
        color: #2c77b4; 
        font-weight: 600; 
        font-size: 20px;
        margin-bottom: 8px;
      }
      footer { 
        text-align: left; 
        color: #333; 
        padding: 15px 0;
        font-size: 12px;
        margin-top: 10px;
      }
      
      .form-group { 
        margin-bottom: 6px !important;
      }
      .form-group label { 
        margin-bottom: 3px !important;
        font-size: 13px;
        color: #333;
      }
      .form-control { 
        height: 36px !important;
        padding: 5px 10px !important;
        font-size: 13px;
      }
      
      /* SHAP图表样式 */
      .shap-plot-container {
        margin-bottom: 20px;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 10px;
        background-color: #f9f9f9;
      }
      .shap-title {
        text-align: center;
        font-weight: bold;
        color: #2c77b4;
        margin-bottom: 10px;
        font-size: 16px;
      }
    "))
  ),
  
  # 添加页边距容器
  div(style = "margin: 10px;",  
    div(class = "navbar-custom", 
        h2("PM Risk Prediction Model for Colorectal Cancer Patients", 
           style = "margin:0; font-size: 20px;")
    ),
    
    fluidRow(
      column(width = 4,
             div(class = "card",
                 div(class = "section-title", "Input Features"),
                 fluidRow(
                   lapply(names(variables), function(feature) {
                     # 获取默认值
                     default_val <- get_default_value(feature)
                     
                     # 创建输入控件
                     if (feature == "gender") {
                       input_control <- selectInput(feature, clean_name(feature), 
                                                   choices = gender_choices,
                                                   selected = as.character(default_val))
                     } else if (feature == "primary.tumor.sites") {
                       input_control <- selectInput(feature, clean_name(feature), 
                                                   choices = site_choices,
                                                   selected = as.character(default_val))
                     } else if (variables[[feature]] %in% c("numeric", "integer")) {
                       input_control <- numericInput(feature, clean_name(feature), 
                                                    value = default_val)
                     } else {
                       # 其他分类变量
                       input_control <- selectInput(feature, clean_name(feature), 
                                                   choices = task_model$levels(feature)[[1]],
                                                   selected = as.character(default_val))
                     }
                     
                     # 返回列
                     column(width = 6, input_control)
                   })
                 ),
                 div(style = "margin-top: 5px;",
                     actionButton("predict", "Predict Now", 
                                  class = "btn-primary", 
                                  style = "width:100%; height: 40px; font-size: 16px; border-radius: 8px;")
                 )
             )
      ),
      
      column(width = 8,
             # 全局SHAP分析卡片
             div(class = "card",
                 div(class = "section-title", "Global SHAP Analysis"),
                 div(class = "shap-plot-container",
                     div(class = "shap-title", "Global SHAP Importance Plot"),
                     plotOutput("global_shap_bar", width = "100%", height = "300px")
                 ),
                 div(class = "shap-plot-container",
                     div(class = "shap-title", "Global SHAP Beeswarm Plot"),
                     plotOutput("global_shap_beeswarm", width = "100%", height = "400px")
                 )
             ),
             
             div(class = "card",
                 div(class = "section-title", "Prediction Result"),
                 uiOutput("prob_text"),
                 div(class = "risk-container", uiOutput("dynamic_indicator")),
                 div(class = "scale-wrapper",
                     span(class = "scale-label", style = "left: 0%; transform: none;", "0 Low Risk"),
                     span(class = "scale-label", style = "left: 30%;", "0.3"),
                     span(class = "scale-label", style = "left: 50%;", "0.5"),
                     span(class = "scale-label", style = "right: 0%; transform: none;", "High Risk 1")
                 ),
                 uiOutput("risk_badge")
             ),
             
             div(class = "card",
                 div(class = "section-title", "Individual SHAP Analysis"),
                 uiOutput("shap_loading_indicator"),
                 plotOutput("waterfall", height = "310px"),
                 div(style = "padding-left: 150px;", plotOutput("force_plot", height = "220px"))
             )
      )
    ),
    
  )
)

# --- 4. Server 逻辑（包含全局SHAP分析）---
server <- function(input, output, session) {
  
  # 响应式值存储
  pred_result <- reactiveVal(NULL)
  shap_result <- reactiveVal(NULL)
  shap_computing <- reactiveVal(FALSE)
  
  # 预加载背景数据（加速个体SHAP计算）
  bg_data_cache <- if (model_loaded && nrow(train_data) > 0) {
    n_bg <- min(30, nrow(train_data))
    train_data[sample(nrow(train_data), n_bg), task_model$feature_names, drop = FALSE]
  } else {
    NULL
  }
  
  # 渲染全局SHAP条形图（重要性图）
  output$global_shap_bar <- renderPlot({
    if (exists("SHAP_sv_ChooseModel")) {
      # 创建自定义主题
      custom_theme <- theme_minimal() +
        theme(
          panel.background = element_rect(fill = "white", color = NA),
          panel.grid.major = element_line(color = "#e0e0e0", linewidth = 0.5),
          panel.grid.minor = element_blank(),
          axis.line = element_line(color = "black", linewidth = 0.5),
          axis.text = element_text(size = 12, color = "black"),
          axis.title = element_text(size = 14, face = "bold", color = "black"),
          plot.title = element_text(size = 16, face = "bold", hjust = 0.5, color = "#2c77b4"),
          legend.position = "none"
        )
      
      # 创建条形图
      sv_importance(SHAP_sv_ChooseModel) + 
        custom_theme +
        labs(
          title = "Global SHAP Feature Importance",
          x = "mean(|SHAP value|)",
          y = "Feature"
        ) +
        scale_fill_manual(values = "#FFA726")  # 橙黄色
    } else {
      # 如果没有全局SHAP对象，显示提示
      plot(1, type = "n", xlab = "", ylab = "", xlim = c(0, 1), ylim = c(0, 1),
           main = "Global SHAP Importance Plot Not Available")
      text(0.5, 0.5, "Global SHAP model not loaded or could not be calculated", 
           cex = 1.2, col = "darkred")
    }
  })
  
  # 渲染全局SHAP蜂群图（散点图）
  output$global_shap_beeswarm <- renderPlot({
    if (exists("SHAP_sv_ChooseModel")) {
      # 创建自定义主题
      custom_theme <- theme_minimal() +
        theme(
          panel.background = element_rect(fill = "#f5f5f5", color = NA),  # 灰色背景
          panel.grid.major = element_line(color = "#d0d0d0", linewidth = 0.5),
          panel.grid.minor = element_blank(),
          axis.line = element_line(color = "black", linewidth = 0.5),
          axis.text = element_text(size = 12, color = "black"),
          axis.title = element_text(size = 14, face = "bold", color = "black"),
          plot.title = element_text(size = 16, face = "bold", hjust = 0.5, color = "#2c77b4"),
          legend.position = "right",
          legend.title = element_text(size = 12, face = "bold"),
          legend.text = element_text(size = 11)
        )
      
      # 创建蜂群图
      sv_importance(SHAP_sv_ChooseModel, kind = "beeswarm", show_numbers = FALSE) + 
        custom_theme +
        labs(
          title = "Global SHAP Beeswarm Plot",
          x = "SHAP Value",
          y = "Feature"
        ) +
        scale_color_gradient2(
          low = "#2166ac",  # 蓝色代表低值
          mid = "#f7f7f7",  # 白色代表中间值
          high = "#b2182b", # 红色代表高值
          midpoint = 0,
          name = "Feature Value",
          labels = c("Low", "", "High"),
          breaks = c(-3, 0, 3)  # 根据您的数据调整
        )
    } else {
      # 如果没有全局SHAP对象，显示提示
      plot(1, type = "n", xlab = "", ylab = "", xlim = c(0, 1), ylim = c(0, 1),
           main = "Global SHAP Beeswarm Plot Not Available")
      text(0.5, 0.5, "Global SHAP model not loaded or could not be calculated", 
           cex = 1.2, col = "darkred")
    }
  })
  
  # 加载指示器
  output$shap_loading_indicator <- renderUI({
    if (shap_computing()) {
      div(style = "text-align: center; padding: 20px; color: #2c77b4; font-weight: bold;",
          "⏳ Computing individual SHAP values... (This may take 30-60 seconds)")
    }
  })
  
  observeEvent(input$predict, {
    shap_computing(TRUE)  # 标记SHAP计算开始
    
    # 准备输入数据
    input_list <- lapply(names(variables), function(f) {
      val <- input[[f]]
      if (variables[[f]] %in% c("numeric", "integer")) {
        return(as.numeric(val))
      }
      return(val)
    })
    
    input_df <- as.data.frame(input_list)
    colnames(input_df) <- names(variables)
    
    # 处理因子变量
    for (f in names(variables)) {
      if (variables[[f]] == "factor") {
        input_df[[f]] <- factor(input_df[[f]], levels = task_model$levels(f)[[1]])
      }
    }
    
    # 进行预测（快速）
    pred <- model_ChooseModel_aftertune$predict_newdata(input_df)
    prob <- round(as.numeric(as.data.table(pred)$prob.1), 3)
    pred_result(prob)
    
    # 立即更新预测结果和风险指示器（不等待SHAP）
    output$prob_text <- renderUI({
      div(class = "prob-text-style", paste("The probability that this patient has the disease is", prob))
    })
    
    output$dynamic_indicator <- renderUI({
      pos <- if (prob <= 0.3) {
        (prob / 0.3) * 30
      } else if (prob <= 0.5) {
        30 + ((prob - 0.3) / (0.5 - 0.3)) * 20
      } else {
        50 + ((prob - 0.5) / (1 - 0.5)) * 50
      }
      tags$div(class = "risk-indicator", style = paste0("left: ", max(0, min(100, pos)), "%;"))
    })
    
    output$risk_badge <- renderUI({
      res <- if(prob > 0.5) {
        list("#d9534f", "High Risk")
      } else if(prob >= 0.3) {
        list("#f0ad4e", "Medium Risk")
      } else {
        list("#5cb85c", "Low Risk")
      }
      div(span(class = "risk-label-badge", style = paste0("background-color:", res[[1]]), res[[2]]))
    })
    
    # 异步计算个体SHAP值（不阻塞UI）
    session$onFlushed(function() {
      tryCatch({
        pred_fun <- function(obj, newdata) {
          as.numeric(as.data.table(obj$predict_newdata(newdata))$prob.1)
        }
        
        # 🚀 优化 2：个体SHAP计算 - 减少蒙特卡洛迭代次数
        # 优化参数：背景样本30，蒙特卡洛迭代 50 次（从100改为50，快速 50%）
        # 注意：kernelshap 使用 m 参数而不是 n_samples
        shap_vals <- kernelshap(
          model_ChooseModel_aftertune, 
          input_df,
          bg_X     = bg_data_cache,
          pred_fun = pred_fun,
          m        = 50  # 蒙特卡洛迭代次数（优化：从100改为50，快速 50%）
        )
        
        colnames(shap_vals$S) <- sapply(colnames(shap_vals$S), clean_name_for_plot)
        colnames(shap_vals$X) <- sapply(colnames(shap_vals$X), clean_name_for_plot)
        
        sv_obj <- shapviz(shap_vals)
        shap_result(sv_obj)
        
        theme_clean <- theme_minimal() + 
          theme(panel.grid = element_blank(), 
                panel.border = element_blank(),
                axis.line.x = element_line(color = "black"),
                axis.line.y = element_blank(),
                axis.text = element_text(size = 10),
                axis.title = element_text(size = 10))
        
        output$waterfall <- renderPlot({
          sv_waterfall(sv_obj) + theme_clean + labs(title = "SHAP Waterfall Plot", x = "SHAP Value", y = "")
        })
        
        output$force_plot <- renderPlot({
          sv_force(sv_obj) + theme_clean + 
            theme(axis.line.x = element_line(color = "black"),
                  axis.line.y = element_blank(), 
                  axis.text.y = element_blank()) +
            labs(title = "Individual SHAP Force Plot", x = "Prediction Value", y = "")
        })
      }, error = function(e) {
        warning(paste("SHAP computation failed:", conditionMessage(e)))
      })
      shap_computing(FALSE)  # 标记SHAP计算完成
    }, once = TRUE)
  })
}

shinyApp(ui, server)