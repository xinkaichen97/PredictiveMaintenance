#
# This is a Shiny web application. You can run the application by clicking
# the 'Run App' button above.
#
# Find out more about building applications with Shiny here:
#
#    http://shiny.rstudio.com/
#

library(shiny)
library(shinyjs)
library(ggplot2)
library(plyr)
library(reshape2)
library(scales)
table.data <- read.csv(file = 'ISSUE_RECEIPT_OHQ_BW_v2_TEST.csv', header = TRUE, skipNul = TRUE)
table.data[is.na(table.data)] <- 0
plants <- sort(unique(table.data$ORGANIZATION_CODE))
plants <- c('All Organizations (Aggregate)', 'All Organizations', plants)
items <- sort(unique(table.data$ITEM))
if('#VALUE!' %in% items) items <- items[items != '#VALUE!']
items <- c('All Items (Aggregate)', items)
years <- sort(unique(table.data$YEAR))
years <- c('All Years', years)

# Define UI for application that draws a histogram
ui <- fluidPage(

    # Application title
    h3("Summary of On-hand Quantities, Issues, and Receipts"),
    # tags$br(),
    # Sidebar with a slider input for number of bins
    useShinyjs(),
    fluidRow(
        column(3, wellPanel(
            selectInput(inputId = 'Item', label = 'Item code', choices = items, multiple = TRUE),
            selectInput(inputId = 'Org', label = 'Organization', choices = plants, multiple = TRUE),
            selectInput(inputId = 'Year', label = 'Year', choices = years, multiple = TRUE),
            radioButtons(inputId = 'Level', label = 'Report Level', choices = c('Year', 'Quarter', 'Month', 'Week'), inline = FALSE),
            radioButtons(inputId = 'Show', label = 'Show In', choices = c('Quantity', 'Cost'), inline = FALSE),
            submitButton('Report')
        )),
        column(7, plotOutput("distPlot", width = '130%', height = '570'))
    )
)

# Define server logic required to draw a histogram
server <- function(input, output) {
    output$distPlot <- renderPlot({
        filter.data <- table.data
        if(!'All Items (Aggregate)' %in% input$Item) {
            filter.data <- subset(filter.data, ITEM %in% input$Item)
        }
        if(!'All Years' %in% input$Year) {
            filter.data = subset(filter.data, YEAR %in% input$Year)
        } 
        if('All Organizations (Aggregate)' %in% input$Org) {
            filter.data$ORGANIZATION_CODE = 'All Organizations'
        } else {
            if(!'All Organizations' %in% input$Org) {
                filter.data <- subset(filter.data, ORGANIZATION_CODE %in% input$Org)
            }
        }
        filter.data <- subset(filter.data, GROUP_LEVEL == toupper(input$Level))
        if(nrow(filter.data) == 0) {
            showNotification('No data found. Please change the filter.')
        } else {
            grouped.data <- ddply(filter.data, c('ORGANIZATION_CODE', 'YEAR', 'WMYQ_NUMBER'), summarize, 
                                  Issue = sum(QUANTITY_ISSUE), Issue.Dollar = sum(QUANTITY_ISSUE_VALUE), 
                                  Receipt = sum(QUANTITY_RECEIPT), Receipt.Dollar = sum(QUANTITY_RECEIPT_VALUE), 
                                  On.Hand = sum(ONHAND_QTY), On.Hand.Dollar = sum(ONHAND_QTY_VALUE))
            p <- ggplot()
            if(input$Show == 'Quantity') {
                melted.qty <- melt(data = grouped.data, id.vars = c('ORGANIZATION_CODE', 'YEAR', 'WMYQ_NUMBER'), 
                                   measure.vars = c('Issue', 'Receipt', 'On.Hand'), variable.name = 'Transaction', 
                                   value.name = 'Quantity')
                p <- p +
                    geom_bar(data = subset(melted.qty, Transaction != 'On.Hand'), 
                             mapping = aes(x = factor(WMYQ_NUMBER), y = Quantity, fill = Transaction), width = 0.5,
                             stat = 'identity', position = 'dodge') +
                    geom_point(data = subset(melted.qty, Transaction == 'On.Hand'), 
                             mapping = aes(x = factor(WMYQ_NUMBER), y = Quantity, color = Transaction), shape = 4, size = 2, stroke = 2) +
                    scale_color_manual(name = '', values = c('On.Hand' = '#5499C7'), labels = c('On Hand')) +
                    scale_fill_manual(name = "Transaction", values = c('Issue' = '#F1948A', 'Receipt' = '#76D7C4')) +
                    xlab(input$Level)+
                    coord_flip()
                
                # plot appropriate x axis scales for quantities
                if(max(melted.qty$Quantity) >= 1e6) {
                    p <- p + scale_y_continuous(label = dollar_format(prefix = '', scale = .000001, suffix = "M")) + 
                        theme(axis.text.x = element_text(angle = 30, hjust = 1))
                } else if(max(melted.qty$Quantity) >= 1e3) {
                    p <- p + scale_y_continuous(label = dollar_format(prefix = '', scale = .001, suffix = "K")) + 
                        theme(axis.text.x = element_text(angle = 30, hjust = 1))
                } else {
                    p <- p + scale_y_continuous(label = dollar_format(prefix = ''))
                }
                
            } else {
                melted.dollar <- melt(data = grouped.data, id.vars = c('ORGANIZATION_CODE', 'YEAR', 'WMYQ_NUMBER'), 
                                      measure.vars = c('Issue.Dollar', 'Receipt.Dollar', 'On.Hand.Dollar'), 
                                      variable.name = 'Transaction', value.name = 'Cost')
                p <- p +
                    geom_bar(data = subset(melted.dollar, Transaction != 'On.Hand.Dollar'), 
                             mapping = aes(x = factor(WMYQ_NUMBER), y = Cost, fill = Transaction), width = 0.5,
                             stat = 'identity', position = 'dodge') +
                    geom_point(data = subset(melted.dollar, Transaction == 'On.Hand.Dollar'), 
                               mapping = aes(x = factor(WMYQ_NUMBER), y = Cost, color = Transaction), shape = 4,  size = 2, stroke = 2) +
                    scale_color_manual(name = '', values = c('On.Hand.Dollar' = '#5499C7'), labels = c('On Hand')) +
                    scale_fill_manual(name = "Transaction", values = c('Issue.Dollar' = '#F1948A', 'Receipt.Dollar' = '#76D7C4'),
                                      labels = c('Issue', 'Receipt')) +
                    xlab(input$Level)+
                    coord_flip() 
                # plot appropriate x axis scales for dollars
                if(max(melted.dollar$Cost) >= 1e6) {
                    p <- p + scale_y_continuous(label = dollar_format(scale = .000001, suffix = "M")) + 
                        theme(axis.text.x = element_text(angle = 30, hjust = 1))
                } else if(max(melted.dollar$Cost) >= 1e3) {
                    p <- p + scale_y_continuous(label = dollar_format(scale = .001, suffix = "K")) + 
                        theme(axis.text.x = element_text(angle = 30, hjust = 1))
                } else {
                    p <- p + scale_y_continuous(label = dollar_format())
                }
            }
            # avoid showing redundant year level
            if(input$Level == 'Year') {
                p <- p + facet_grid(ORGANIZATION_CODE ~ .)
            } else {
                p <- p + facet_grid(ORGANIZATION_CODE ~ factor(YEAR))
            }
            org_num = length(unique(grouped.data$ORGANIZATION_CODE))
            if((input$Level == 'Month' & org_num > 5) | 
               (input$Level == 'Week' & org_num > 1)) {
                showNotification('The details may not be clear. 
                                 Consider reduce number of organizations or change to higher report level.')
            }
            plot(p)
        }
    })
}

# Run the application 
shinyApp(ui = ui, server = server)
